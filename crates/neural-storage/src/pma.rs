use std::fmt::Debug;

/// Packed Memory Array (PMA)
///
/// A dynamic array that maintains elements in sorted order with O(log^2 N) insertions
/// and O(1) scans. It uses a "gaps" strategy with dynamic rebalancing based on
/// density thresholds.
///
/// Ref: Bender et al., "An Adaptive Packed-Memory Array"
#[derive(Debug)]
pub struct PackedMemoryArray<T> {
    /// The physical backing store. Gaps are represented by None.
    data: Vec<Option<T>>,
    /// Number of actual elements.
    count: usize,
    /// Size of a leaf segment. Typically log(N).
    segment_size: usize,
}

impl<T: Ord + Clone + Debug> Default for PackedMemoryArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone + Debug> PackedMemoryArray<T> {
    pub fn new() -> Self {
        // Initial capacity 16, segment size 4
        let initial_cap = 16;
        Self {
            data: vec![None; initial_cap],
            count: 0,
            segment_size: 4,
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns the total capacity (including gaps).
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Checks if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Gets a reference to the element at physical index `index`.
    /// Note: This is NOT the logical index. Use iteration for logical access.
    pub fn get_physical(&self, index: usize) -> Option<&T> {
        self.data.get(index).and_then(|x| x.as_ref())
    }

    /// Returns an iterator over the elements in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().filter_map(|x| x.as_ref())
    }

    /// Inserts a value while maintaining sorted order.
    pub fn insert(&mut self, value: T) {
        if self.count == 0 {
            self.data[0] = Some(value);
            self.count = 1;
            return;
        }

        // 1. Find the logical position to insert (Binary Search on existing elements)
        // Since the array has gaps, we can't do a standard binary search on indices.
        // We simulate it or find the physical index of the first element >= value.
        // For MVP, we'll do a linear scan or a collected binary search.
        // Optimization: In a real PMA, we can do binary search using an auxiliary index
        // or smart skipping. Here we collect references for simplicity of the "find" step
        // (Performance note: this scan is O(N) in this naive impl, but PMA focus is on the shift/rebalance cost).
        // To be strict O(log N) find, we need the implicit tree.
        // For this sprint, let's focus on the rebalancing mechanics.
        
        // Find physical index where `value` should go.
        // We are looking for the first `Some(x)` where `x >= value`.
        let mut insert_pos = self.data.len(); 
        for (i, slot) in self.data.iter().enumerate() {
            if let Some(x) = slot {
                if *x >= value {
                    insert_pos = i;
                    break;
                }
            }
        }
        
        // If we didn't find a larger element, insert at the end of the data (after last element)
        // But "end of data" might not be `data.len()`. It's after the last Some.
        if insert_pos == self.data.len() {
             // Find last occupied slot
             insert_pos = self.find_last_occupied().map(|i| i + 1).unwrap_or(0);
        }

        // 2. Insert at `insert_pos`. This might require shifting or rebalancing.
        self.insert_at_physical(insert_pos, value);
    }

    fn find_last_occupied(&self) -> Option<usize> {
        self.data.iter().rposition(|x| x.is_some())
    }

    /// Physical insertion with PMA rebalancing logic.
    fn insert_at_physical(&mut self, mut index: usize, value: T) {
        // Ensure valid index
        if index > self.data.len() {
            index = self.data.len();
        }

        // If the specific slot is empty, just put it there? 
        // No, we need to maintain sorted order. If we insert at `index`, 
        // we implicitly assume `index` is the correct sorted position.
        // If `index` is occupied, we shift right into a gap.
        
        // Check if we need to resize global array first (amortized O(1))
        // Standard PMA resize condition: if density > P_max_root
        if self.count as f64 / self.data.len() as f64 > 0.75 { // simplistic max density
            self.resize(self.data.len() * 2);
            // After resize, indices change because we spread elements. 
            // We need to re-find the insert position.
            // For MVP simplicity, let's just recursively call insert (logical find).
            self.insert(value);
            return;
        }

        // Try to shift locally in the segment
        // Segment index
        let segment_idx = index / self.segment_size;
        let segment_start = segment_idx * self.segment_size;
        let segment_end = std::cmp::min(segment_start + self.segment_size, self.data.len());

        // Check density of this segment
        // If we can just shift elements in this segment to open a spot, great.
        // If segment is full, we look up the tree.

        // Simulating the "Window" search
        let mut window_start = segment_start;
        let mut window_end = segment_end;
        let mut window_size = self.segment_size;
        let mut height = 0;

        loop {
            // Count elements in window
            let window_count = self.count_range(window_start, window_end);
            let capacity = window_end - window_start;
            let density = (window_count + 1) as f64 / capacity as f64; // +1 for the new element

            // Thresholds for height `height`
            // Root is height H. Leaves height 0.
            // Simplified thresholds:
            // Upper bound T_h.
            // T_0 (leaves) = 1.0 (strict? No, typically 0.92 or similar, but let's say we allow 1.0 locally if we split?)
            // Actually PMA allows leaves to be full IF we can rebalance parent.
            // Let's use: T_h = T_0 + (T_root - T_0) * (h / H)
            // But for this MVP, let's just check: "Can we fit it?"
            
            if density <= self.upper_threshold(height, capacity) {
                // Found a window that accepts the new density!
                // Rebalance this window: Spread elements evenly including the new one.
                self.rebalance_window(window_start, window_end, index, value);
                self.count += 1;
                return;
            }

            // If window is full (violates density), grow window (go to parent)
            // Implicit tree: parent of a segment-aligned window of size S is window of size 2S aligned to 2S.
            let next_size = window_size * 2;
            let new_start = (window_start / next_size) * next_size;
            let new_end = std::cmp::min(new_start + next_size, self.data.len());
            
            if new_start == 0 && new_end == self.data.len() {
                // Reached root and it's full (should have been caught by global resize, but just in case)
                self.resize(self.data.len() * 2);
                self.insert(value);
                return;
            }

            window_start = new_start;
            window_end = new_end;
            window_size = next_size;
            height += 1;
        }
    }

    fn count_range(&self, start: usize, end: usize) -> usize {
        self.data[start..end].iter().filter(|x| x.is_some()).count()
    }

    /// Upper density threshold for a given level
    fn upper_threshold(&self, height: usize, _capacity: usize) -> f64 {
        // Simple progression
        // Height 0 (leaves): 1.0 (allow full if we rebalance?)
        // Actually, strictly < 1.0 helps performance, but 1.0 is hard constraint.
        // Let's use:
        // Level 0: 0.9
        // Root: 0.75 (wait, root is typically less dense to allow cheap inserts?)
        // Standard PMA:
        // T_root < T_leaf. e.g. Root 0.5, Leaf 1.0?
        // No, root needs to be sparse to allow growth? 
        // Wait, if root is 50% full, we double. So max density at root is 0.5 (or 0.75).
        // Let's assume Root max density = 0.75.
        // Leaf max density = 1.0.
        // Interpolate.
        
        let root_density = 0.75;
        let leaf_density = 1.0;
        
        // We need total height H. 
        // H = log2(N / segment_size)
        let total_segments = (self.data.len() as f64 / self.segment_size as f64).ceil();
        let max_height = total_segments.log2().ceil() as usize;
        
        if max_height == 0 { return leaf_density; }

        let delta = (root_density - leaf_density) / max_height as f64;
        leaf_density + (delta * height as f64)
    }

    /// Spreads elements in [start, end) evenly, inserting `val` at logical `insert_pos` relative to window elements.
    /// Note: `physical_insert_pos` was the target, but during rebalance we just collect all elements, add the new one, and redistribute.
    fn rebalance_window(&mut self, start: usize, end: usize, _target_idx: usize, value: T) {
        // 1. Collect all elements in this window
        let mut elements: Vec<T> = self.data[start..end]
            .iter_mut()
            .filter_map(|x| x.take())
            .collect();
        
        // 2. Insert the new value into the sorted list of elements
        // Since `elements` is sorted, we can use binary search
        let pos = elements.binary_search(&value).unwrap_or_else(|e| e);
        elements.insert(pos, value);

        // 3. Redistribute evenly
        let count = elements.len();
        let capacity = end - start;
        
        // Simple even spacing
        // e.g. Cap 8, Count 3. Gaps 5.
        // Spacing factor = Cap / Count
        
        let spacing = capacity as f64 / count as f64;
        
        for (i, item) in elements.into_iter().enumerate() {
            let target_pos = start + (i as f64 * spacing) as usize;
            self.data[target_pos] = Some(item);
        }
    }

    fn resize(&mut self, new_cap: usize) {
        // Collect all elements
        let elements: Vec<T> = self.data.iter_mut().filter_map(|x| x.take()).collect();
        
        // Create new buffer
        self.data = vec![None; new_cap];
        self.segment_size = (new_cap as f64).log2().ceil() as usize; // Update segment size? Or keep fixed?
        // Usually segment size grows slowly or stays fixed (page size). Let's keep it somewhat related to log N.
        if self.segment_size < 4 { self.segment_size = 4; }

        // Spread elements
        let count = elements.len();
        let spacing = new_cap as f64 / count as f64;
        
        for (i, item) in elements.into_iter().enumerate() {
            let pos = (i as f64 * spacing) as usize;
            self.data[pos] = Some(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insertion() {
        let mut pma = PackedMemoryArray::new();
        pma.insert(10);
        pma.insert(5);
        pma.insert(20);
        pma.insert(15);

        let values: Vec<i32> = pma.iter().cloned().collect();
        assert_eq!(values, vec![5, 10, 15, 20]);
    }

    #[test]
    fn test_rebalance_trigger() {
        let mut pma = PackedMemoryArray::new();
        // Insert enough to trigger local segment fills and window rebalances
        for i in (0..50).rev() {
            pma.insert(i);
        }

        let values: Vec<i32> = pma.iter().cloned().collect();
        assert_eq!(values.len(), 50);
        
        // Check sortedness
        for window in values.windows(2) {
            assert!(window[0] < window[1]);
        }
        
        // Check gaps exist (capacity should be > count)
        assert!(pma.capacity() > pma.len());
    }

    #[test]
    fn test_interleaved_insertions() {
        let mut pma = PackedMemoryArray::new();
        pma.insert(1);
        pma.insert(100);
        pma.insert(50);
        pma.insert(25);
        pma.insert(75);
        
        let values: Vec<i32> = pma.iter().cloned().collect();
        assert_eq!(values, vec![1, 25, 50, 75, 100]);
    }
}
