//! Arrow Flight Server for NeuralGraphDB
//!
//! Enables high-performance, zero-copy data transfer of graph data.

use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, StringArray, StringBuilder, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_flight::{
    encode::FlightDataEncoderBuilder,
    flight_service_server::FlightService,
    Action, Criteria, Empty, FlightData, FlightDescriptor, FlightEndpoint,
    FlightInfo, HandshakeRequest, HandshakeResponse, Ticket, PollInfo,
};
use futures::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use neural_storage::GraphStore;
use neural_core::Graph;
use neural_executor::{QueryResult, Value, Row};

/// Converts a QueryResult to an Arrow RecordBatch
fn query_result_to_record_batch(result: &QueryResult) -> Result<RecordBatch, String> {
    let columns = result.columns();
    let rows = result.rows();
    let row_count = result.row_count();

    if row_count == 0 {
        let fields: Vec<Field> = columns
            .iter()
            .map(|c| Field::new(c, DataType::Utf8, true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        return Ok(RecordBatch::new_empty(schema));
    }

    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());

    for col_name in columns {
        // Infer type from all non-null values to detect mixed types
        let inferred_type = infer_column_type(col_name, rows);
        fields.push(Field::new(col_name, inferred_type.clone(), true));

        let array: ArrayRef = build_array_for_type(col_name, rows, &inferred_type, row_count)?;
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| format!("Failed to create RecordBatch: {}", e))
}

/// Infers the most appropriate data type for a column
fn infer_column_type(col_name: &str, rows: &[Row]) -> DataType {
    let mut has_int = false;
    let mut has_float = false;
    let mut has_bool = false;
    let mut has_other = false;

    for row in rows {
        if let Some(val) = row.get(col_name) {
            match val {
                Value::Null => continue,
                Value::Int(_) => has_int = true,
                Value::Float(_) => has_float = true,
                Value::Bool(_) => has_bool = true,
                _ => has_other = true,
            }
        }
    }

    // Prioritize type coercion: Float > Int > String
    // If there's any mix of numeric types, use Float64
    if has_float || (has_int && has_float) { // Fixed logic: if float exists, or int+float mix
         DataType::Float64
    } else if has_int && !has_bool && !has_other && !has_float {
        DataType::Int64
    } else if has_bool && !has_int && !has_float && !has_other {
        DataType::Boolean
    } else {
        DataType::Utf8
    }
}

/// Builds an Arrow array for the given column and data type
fn build_array_for_type(
    col_name: &str,
    rows: &[Row],
    data_type: &DataType,
    row_count: usize,
) -> Result<ArrayRef, String> {
    match data_type {
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(row_count);
            for row in rows {
                match row.get(col_name) {
                    Some(Value::Int(v)) => builder.append_value(*v),
                    Some(Value::Null) | None => builder.append_null(),
                    Some(v) => return Err(format!("Expected Int64, found {:?} in column {}", v, col_name)),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(row_count);
            for row in rows {
                match row.get(col_name) {
                    Some(Value::Float(v)) => builder.append_value(*v),
                    Some(Value::Int(v)) => builder.append_value(*v as f64), // Coerce int to float
                    Some(Value::Null) | None => builder.append_null(),
                    Some(v) => return Err(format!("Expected Float64, found {:?} in column {}", v, col_name)),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Boolean => {
            let mut builder = BooleanBuilder::with_capacity(row_count);
            for row in rows {
                match row.get(col_name) {
                    Some(Value::Bool(v)) => builder.append_value(*v),
                    Some(Value::Null) | None => builder.append_null(),
                    Some(v) => return Err(format!("Expected Boolean, found {:?} in column {}", v, col_name)),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        _ => {
            // Default to string representation
            let mut builder = StringBuilder::with_capacity(row_count, row_count * 20);
            for row in rows {
                match row.get(col_name) {
                    Some(Value::Null) | None => builder.append_null(),
                    Some(Value::String(s)) => builder.append_value(s),
                    Some(Value::Node(id)) => builder.append_value(format!("Node({})", id)),
                    Some(Value::Int(v)) => builder.append_value(v.to_string()),
                    Some(Value::Float(v)) => builder.append_value(v.to_string()),
                    Some(Value::Bool(v)) => builder.append_value(v.to_string()),
                    Some(val) => builder.append_value(format!("{:?}", val)),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
    }
}

/// Implementation of the Arrow Flight service
pub struct NeuralFlightService {
    store: Arc<GraphStore>,
}

impl NeuralFlightService {
    pub fn new(store: Arc<GraphStore>) -> Self {
        Self { store }
    }

    /// Helper to create the nodes schema
    fn nodes_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("label", DataType::Utf8, true),
            Field::new("properties_json", DataType::Utf8, true),
        ]))
    }

    /// Helper to create the edges schema
    fn edges_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("source", DataType::UInt64, false),
            Field::new("target", DataType::UInt64, false),
            Field::new("type", DataType::Utf8, true),
        ]))
    }
}

#[tonic::async_trait]
impl FlightService for NeuralFlightService {
    type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>;
    type ListFlightsStream = Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send>>;
    type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;
    type DoPutStream = Pin<Box<dyn Stream<Item = Result<arrow_flight::PutResult, Status>> + Send>>;
    type DoActionStream = Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send>>;
    type ListActionsStream = Pin<Box<dyn Stream<Item = Result<arrow_flight::ActionType, Status>> + Send>>;
    type DoExchangeStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;

    async fn handshake(
        &self,
        _request: Request<tonic::Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let path = descriptor.path.first().cloned().unwrap_or_default();

        let schema = match path.as_str() {
            "nodes" => Self::nodes_schema(),
            "edges" => Self::edges_schema(),
            _ => return Err(Status::not_found(format!("Flight not found: {}", path))),
        };

        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(descriptor)
            .with_endpoint(FlightEndpoint::new().with_ticket(Ticket::new(path)));

        Ok(Response::new(info))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<arrow_flight::SchemaResult>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let query_str = std::str::from_utf8(&ticket.ticket)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        match query_str {
            "nodes" => {
                let schema = Self::nodes_schema();
                let store = self.store.clone();

                let mut ids = Vec::new();
                let mut labels = Vec::new();
                let mut props = Vec::new();

                for i in 0..store.node_count() {
                    let node_id = neural_core::NodeId::new(i as u64);
                    ids.push(i as u64);
                    labels.push(store.get_label(node_id).map(|s| s.to_string()));
                    // Minimal prop serialization for now
                    let mut p_map = std::collections::HashMap::new();
                    p_map.insert("id", format!("{}", i));
                    props.push(serde_json::to_string(&p_map).unwrap());
                }

                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(UInt64Array::from(ids)),
                        Arc::new(StringArray::from(labels)),
                        Arc::new(StringArray::from(props)),
                    ],
                )
                .map_err(|e| Status::internal(e.to_string()))?;

                let batches = vec![batch];
                let stream = FlightDataEncoderBuilder::new()
                    .with_schema(schema)
                    .build(futures::stream::iter(batches).map(Ok))
                    .map(|res| res.map_err(|e| Status::internal(e.to_string())));

                Ok(Response::new(Box::pin(stream)))
            }
            "edges" => {
                 let schema = Self::edges_schema();
                 let store = self.store.clone();

                 let mut sources = Vec::new();
                 let mut targets = Vec::new();
                 let mut types: Vec<Option<String>> = Vec::new();

                 for i in 0..store.node_count() {
                     let u = neural_core::NodeId::new(i as u64);
                     for v in store.neighbors(u) {
                         sources.push(i as u64);
                         targets.push(v.as_u64());
                         types.push(None); 
                     }
                 }

                 let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(UInt64Array::from(sources)),
                        Arc::new(UInt64Array::from(targets)),
                        Arc::new(StringArray::from(types)),
                    ],
                )
                .map_err(|e| Status::internal(e.to_string()))?;

                let batches = vec![batch];
                let stream = FlightDataEncoderBuilder::new()
                    .with_schema(schema)
                    .build(futures::stream::iter(batches).map(Ok))
                    .map(|res| res.map_err(|e| Status::internal(e.to_string())));

                Ok(Response::new(Box::pin(stream)))
            }
            query => {
                // Execute NGQL query
                let store = self.store.clone();
                let query = query.to_string();
                
                // We need to execute strictly; Flight is read-heavy usually, but we allow all
                let result = neural_executor::execute_query(&store, &query)
                    .map_err(|e| Status::internal(format!("Query execution failed: {}", e)))?;
                
                // Convert QueryResult to RecordBatch
                let batch = query_result_to_record_batch(&result)
                    .map_err(|e| Status::internal(format!("Arrow conversion failed: {}", e)))?;
                
                let schema = batch.schema();
                let batches = vec![batch];
                let stream = FlightDataEncoderBuilder::new()
                    .with_schema(schema)
                    .build(futures::stream::iter(batches).map(Ok))
                    .map(|res| res.map_err(|e| Status::internal(e.to_string())));

                Ok(Response::new(Box::pin(stream)))
            }
        }
    }

    async fn do_put(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_exchange(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_storage::GraphStore;

    #[tokio::test]
    async fn test_nodes_schema() {
        let schema = NeuralFlightService::nodes_schema();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.fields()[0].name(), "id");
    }

    #[tokio::test]
    async fn test_edges_schema() {
        let schema = NeuralFlightService::edges_schema();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.fields()[0].name(), "source");
    }

    #[tokio::test]
    async fn test_get_flight_info() {
        let store = Arc::new(GraphStore::builder().build());
        let service = NeuralFlightService::new(store);
        
        let descriptor = FlightDescriptor::new_path(vec!["nodes".to_string()]);
        let request = Request::new(descriptor);
        
        let response = service.get_flight_info(request).await.unwrap();
        let info = response.into_inner();
        
        assert_eq!(info.endpoint.len(), 1);
    }

    #[tokio::test]
    async fn test_do_get_nodes() {
        let store = Arc::new(GraphStore::builder().add_node(0u64, Vec::<(String, String)>::new()).build());
        let service = NeuralFlightService::new(store);
        
        let ticket = Ticket::new("nodes");
        let request = Request::new(ticket);
        
        let response = service.do_get(request).await.unwrap();
        let stream = response.into_inner();
        let results: Vec<_> = stream.collect().await;
        
        // Should have Schema message + 1 data message
        assert_eq!(results.len(), 2);
    }
}