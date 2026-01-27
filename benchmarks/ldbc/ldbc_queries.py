"""
LDBC-SNB Query Definitions for NeuralGraphDB Benchmark

This module defines the 14 LDBC-SNB Interactive queries adapted for NeuralGraphDB.
The queries are divided into:
- Interactive Short (IS1-IS7): Simple read queries
- Interactive Complex (IC1-IC7): Multi-hop traversal queries

Reference: https://ldbcouncil.org/benchmarks/snb/
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import random


@dataclass
class LDBCQuery:
    """LDBC Query definition."""
    id: str
    name: str
    description: str
    cypher_template: str
    category: str  # 'IS' (Interactive Short) or 'IC' (Interactive Complex)
    params_generator: Optional[str] = None  # Name of method to generate params


# =============================================================================
# Interactive Short Queries (IS1-IS7)
# =============================================================================

IS1_PROFILE = LDBCQuery(
    id="IS1",
    name="Profile",
    description="Given a Person, retrieve their profile including firstName, lastName, birthday, locationIP, browserUsed, gender, creationDate.",
    cypher_template="""
        MATCH (p:Person)
        WHERE p.id = $personId
        RETURN p.firstName, p.lastName, p.birthday, p.locationIP, p.browserUsed, p.gender, p.creationDate
    """,
    category="IS",
    params_generator="person_id"
)

IS2_RECENT_MESSAGES = LDBCQuery(
    id="IS2",
    name="Recent Messages",
    description="Given a Person, retrieve the 10 most recent Messages created by that Person.",
    cypher_template="""
        MATCH (p:Person)<-[:HAS_CREATOR]-(m:Message)
        WHERE p.id = $personId
        RETURN m.id, m.content, m.creationDate
        ORDER BY m.creationDate DESC
        LIMIT 10
    """,
    category="IS",
    params_generator="person_id"
)

IS3_FRIENDS = LDBCQuery(
    id="IS3",
    name="Friends",
    description="Given a Person, retrieve all Persons that they know.",
    cypher_template="""
        MATCH (p:Person)-[:KNOWS]->(friend:Person)
        WHERE p.id = $personId
        RETURN friend.id, friend.firstName, friend.lastName
    """,
    category="IS",
    params_generator="person_id"
)

IS4_MESSAGE_CONTENT = LDBCQuery(
    id="IS4",
    name="Message Content",
    description="Given a Message, retrieve its content and creation date.",
    cypher_template="""
        MATCH (m:Message)
        WHERE m.id = $messageId
        RETURN m.content, m.creationDate, m.imageFile
    """,
    category="IS",
    params_generator="message_id"
)

IS5_MESSAGE_CREATOR = LDBCQuery(
    id="IS5",
    name="Message Creator",
    description="Given a Message, retrieve the Person that created it.",
    cypher_template="""
        MATCH (m:Message)-[:HAS_CREATOR]->(p:Person)
        WHERE m.id = $messageId
        RETURN p.id, p.firstName, p.lastName
    """,
    category="IS",
    params_generator="message_id"
)

IS6_FORUM_OF_MESSAGE = LDBCQuery(
    id="IS6",
    name="Forum of Message",
    description="Given a Message, retrieve the Forum it belongs to.",
    cypher_template="""
        MATCH (m:Message)<-[:CONTAINER_OF]-(f:Forum)
        WHERE m.id = $messageId
        RETURN f.id, f.title
    """,
    category="IS",
    params_generator="message_id"
)

IS7_MESSAGE_REPLIES = LDBCQuery(
    id="IS7",
    name="Message Replies",
    description="Given a Message, retrieve all replies and their authors.",
    cypher_template="""
        MATCH (m:Message)<-[:REPLY_OF]-(reply:Message)-[:HAS_CREATOR]->(p:Person)
        WHERE m.id = $messageId
        RETURN reply.id, reply.content, reply.creationDate, p.id, p.firstName, p.lastName
        ORDER BY reply.creationDate DESC
    """,
    category="IS",
    params_generator="message_id"
)


# =============================================================================
# Interactive Complex Queries (IC1-IC7)
# =============================================================================

IC1_FRIENDS_WITH_NAME = LDBCQuery(
    id="IC1",
    name="Friends with Certain Name",
    description="Given a Person and a name, find friends and friends of friends with that first name.",
    cypher_template="""
        MATCH (p:Person)-[:KNOWS*1..2]->(friend:Person)
        WHERE p.id = $personId AND friend.firstName = $firstName
        RETURN DISTINCT friend.id, friend.lastName, friend.birthday
        ORDER BY friend.lastName
        LIMIT 20
    """,
    category="IC",
    params_generator="person_and_name"
)

IC2_RECENT_MESSAGES_FROM_FRIENDS = LDBCQuery(
    id="IC2",
    name="Recent Messages from Friends",
    description="Given a Person and a date, find the most recent Messages from their friends before that date.",
    cypher_template="""
        MATCH (p:Person)-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(m:Message)
        WHERE p.id = $personId AND m.creationDate < $maxDate
        RETURN friend.id, friend.firstName, friend.lastName, m.id, m.content, m.creationDate
        ORDER BY m.creationDate DESC
        LIMIT 20
    """,
    category="IC",
    params_generator="person_and_date"
)

IC3_FRIENDS_IN_COUNTRIES = LDBCQuery(
    id="IC3",
    name="Friends and Friends of Friends in Countries",
    description="Given a Person and two country names, find friends and friends of friends located in those countries.",
    cypher_template="""
        MATCH (p:Person)-[:KNOWS*1..2]->(friend:Person)-[:IS_LOCATED_IN]->(city:Place)-[:IS_PART_OF]->(country:Place)
        WHERE p.id = $personId AND country.name IN [$countryX, $countryY]
        RETURN DISTINCT friend.id, friend.firstName, friend.lastName, country.name
        LIMIT 20
    """,
    category="IC",
    params_generator="person_and_countries"
)

IC4_NEW_TOPICS = LDBCQuery(
    id="IC4",
    name="New Topics",
    description="Given a Person and a date range, find Tags associated with their friends' posts.",
    cypher_template="""
        MATCH (p:Person)-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag)
        WHERE p.id = $personId AND post.creationDate >= $startDate AND post.creationDate < $endDate
        RETURN tag.name, count(post) AS postCount
        ORDER BY postCount DESC
        LIMIT 10
    """,
    category="IC",
    params_generator="person_and_date_range"
)

IC5_FRIENDS_OF_FRIENDS_POSTS = LDBCQuery(
    id="IC5",
    name="New Groups",
    description="Given a Person, find Forums their friends have joined after a given date.",
    cypher_template="""
        MATCH (p:Person)-[:KNOWS*1..2]->(friend:Person)-[:HAS_MEMBER]->(forum:Forum)
        WHERE p.id = $personId
        WITH forum, count(DISTINCT friend) AS friendCount
        ORDER BY friendCount DESC
        LIMIT 20
        MATCH (forum)-[:CONTAINER_OF]->(post:Post)
        RETURN forum.title, friendCount, count(post) AS postCount
        ORDER BY friendCount DESC, postCount DESC
    """,
    category="IC",
    params_generator="person_id"
)

IC6_TAG_CO_OCCURRENCE = LDBCQuery(
    id="IC6",
    name="Tag Co-occurrence",
    description="Given a Person and a Tag, find other Tags that appear on Posts together with that Tag.",
    cypher_template="""
        MATCH (p:Person)-[:KNOWS*1..2]->(friend:Person)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(knownTag:Tag)
        WHERE p.id = $personId AND knownTag.name = $tagName
        MATCH (post)-[:HAS_TAG]->(otherTag:Tag)
        WHERE otherTag.name <> $tagName
        RETURN otherTag.name, count(post) AS postCount
        ORDER BY postCount DESC
        LIMIT 10
    """,
    category="IC",
    params_generator="person_and_tag"
)

IC7_RECENT_LIKERS = LDBCQuery(
    id="IC7",
    name="Recent Likers",
    description="Given a Person, find Persons who have liked their Messages most recently.",
    cypher_template="""
        MATCH (p:Person)<-[:HAS_CREATOR]-(m:Message)<-[l:LIKES]-(liker:Person)
        WHERE p.id = $personId
        WITH liker, m, l
        ORDER BY l.creationDate DESC
        RETURN DISTINCT liker.id, liker.firstName, liker.lastName
        LIMIT 10
    """,
    category="IC",
    params_generator="person_id"
)


# =============================================================================
# All Queries Collection
# =============================================================================

LDBC_QUERIES: Dict[str, LDBCQuery] = {
    # Interactive Short
    "IS1": IS1_PROFILE,
    "IS2": IS2_RECENT_MESSAGES,
    "IS3": IS3_FRIENDS,
    "IS4": IS4_MESSAGE_CONTENT,
    "IS5": IS5_MESSAGE_CREATOR,
    "IS6": IS6_FORUM_OF_MESSAGE,
    "IS7": IS7_MESSAGE_REPLIES,
    # Interactive Complex
    "IC1": IC1_FRIENDS_WITH_NAME,
    "IC2": IC2_RECENT_MESSAGES_FROM_FRIENDS,
    "IC3": IC3_FRIENDS_IN_COUNTRIES,
    "IC4": IC4_NEW_TOPICS,
    "IC5": IC5_FRIENDS_OF_FRIENDS_POSTS,
    "IC6": IC6_TAG_CO_OCCURRENCE,
    "IC7": IC7_RECENT_LIKERS,
}


class QueryParameterGenerator:
    """Generates query parameters based on loaded data."""

    def __init__(self, persons: List[int], messages: List[int],
                 first_names: List[str], tags: List[str]):
        self.persons = persons
        self.messages = messages
        self.first_names = first_names
        self.tags = tags

    def person_id(self) -> Dict[str, Any]:
        return {"personId": random.choice(self.persons)}

    def message_id(self) -> Dict[str, Any]:
        return {"messageId": random.choice(self.messages)} if self.messages else {"messageId": 0}

    def person_and_name(self) -> Dict[str, Any]:
        return {
            "personId": random.choice(self.persons),
            "firstName": random.choice(self.first_names) if self.first_names else "Alice"
        }

    def person_and_date(self) -> Dict[str, Any]:
        return {
            "personId": random.choice(self.persons),
            "maxDate": "2026-01-01T00:00:00"
        }

    def person_and_countries(self) -> Dict[str, Any]:
        return {
            "personId": random.choice(self.persons),
            "countryX": "United States",
            "countryY": "Germany"
        }

    def person_and_date_range(self) -> Dict[str, Any]:
        return {
            "personId": random.choice(self.persons),
            "startDate": "2020-01-01T00:00:00",
            "endDate": "2026-01-01T00:00:00"
        }

    def person_and_tag(self) -> Dict[str, Any]:
        return {
            "personId": random.choice(self.persons),
            "tagName": random.choice(self.tags) if self.tags else "AI"
        }

    def get_params(self, generator_name: str) -> Dict[str, Any]:
        """Get parameters using the named generator."""
        generator = getattr(self, generator_name, None)
        if generator:
            return generator()
        return {}
