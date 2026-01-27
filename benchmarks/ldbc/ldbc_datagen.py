"""
LDBC-SNB Data Generator for NeuralGraphDB Benchmark

Generates synthetic social network data following LDBC-SNB schema:
- Person nodes with realistic attributes
- KNOWS relationships (social network)
- Message/Post nodes with content
- Forum containers
- Tag associations

Scale Factors:
- SF1:  ~10K persons, ~180K edges
- SF10: ~100K persons, ~1.8M edges
- SF100: ~1M persons, ~18M edges
"""

import csv
import os
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm

# Try to use Faker for realistic data, fallback to simple generation
try:
    from faker import Faker
    fake = Faker()
    Faker.seed(42)
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False
    fake = None

random.seed(42)


@dataclass
class ScaleFactor:
    """LDBC Scale Factor definition."""
    name: str
    num_persons: int
    avg_friends: int
    num_forums: int
    posts_per_forum: int
    num_tags: int

    @property
    def estimated_edges(self) -> int:
        return self.num_persons * self.avg_friends


# Standard LDBC Scale Factors
SCALE_FACTORS = {
    "SF0.1": ScaleFactor("SF0.1", 1_000, 15, 100, 20, 50),
    "SF1": ScaleFactor("SF1", 10_000, 18, 1_000, 30, 100),
    "SF10": ScaleFactor("SF10", 100_000, 18, 10_000, 30, 200),
    "SF100": ScaleFactor("SF100", 1_000_000, 18, 50_000, 40, 500),
}


# Realistic first names for query parameters
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry",
    "Isabella", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zach", "Anna", "Brian", "Chloe", "Daniel"
]

# Common tags for social network posts
COMMON_TAGS = [
    "AI", "MachineLearning", "DeepLearning", "Python", "Rust", "Graph",
    "Database", "Cloud", "DevOps", "Security", "Web", "Mobile", "IoT",
    "Blockchain", "Crypto", "Startup", "Innovation", "Research", "Science",
    "Technology", "Data", "Analytics", "BigData", "ML", "NLP", "ComputerVision"
]

# Countries for location
COUNTRIES = [
    "United States", "Germany", "France", "United Kingdom", "Japan",
    "China", "India", "Brazil", "Canada", "Australia"
]

# Cities by country
CITIES = {
    "United States": ["New York", "San Francisco", "Los Angeles", "Seattle", "Austin"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne"],
    "France": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice"],
    "United Kingdom": ["London", "Manchester", "Birmingham", "Edinburgh", "Bristol"],
    "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya"],
    "China": ["Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Hangzhou"],
    "India": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"],
    "Brazil": ["Sao Paulo", "Rio de Janeiro", "Brasilia", "Salvador", "Belo Horizonte"],
    "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
}


class LDBCDataGenerator:
    """Generates LDBC-SNB compatible data."""

    def __init__(self, scale_factor: str = "SF1", output_dir: str = "benchmarks/ldbc/data"):
        if scale_factor not in SCALE_FACTORS:
            raise ValueError(f"Unknown scale factor: {scale_factor}. Use one of {list(SCALE_FACTORS.keys())}")

        self.sf = SCALE_FACTORS[scale_factor]
        self.output_dir = Path(output_dir) / scale_factor
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for generated IDs
        self.person_ids: List[int] = []
        self.message_ids: List[int] = []
        self.forum_ids: List[int] = []
        self.tag_ids: List[int] = []
        self.first_names_used: List[str] = []
        self.tags_used: List[str] = []

    def generate_all(self) -> "LDBCDataGenerator":
        """Generate all data files."""
        print(f"\n{'='*60}")
        print(f"LDBC-SNB Data Generator - {self.sf.name}")
        print(f"{'='*60}")
        print(f"Target: {self.sf.num_persons:,} persons, ~{self.sf.estimated_edges:,} edges")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        self._generate_tags()
        self._generate_places()
        self._generate_persons()
        self._generate_knows()
        self._generate_forums()
        self._generate_posts()
        self._generate_likes()
        self._write_metadata()

        print(f"\n{'='*60}")
        print(f"Data generation complete!")
        print(f"{'='*60}\n")

        return self

    def _generate_tags(self):
        """Generate Tag nodes."""
        print("Generating Tags...")
        filepath = self.output_dir / "tag.csv"

        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "url"])

            for i in range(self.sf.num_tags):
                if i < len(COMMON_TAGS):
                    name = COMMON_TAGS[i]
                else:
                    name = f"Tag_{i}"
                self.tag_ids.append(i)
                self.tags_used.append(name)
                writer.writerow([i, name, f"https://dbpedia.org/resource/{name}"])

        print(f"  Generated {len(self.tag_ids)} tags")

    def _generate_places(self):
        """Generate Place nodes (Countries and Cities)."""
        print("Generating Places...")
        filepath = self.output_dir / "place.csv"

        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "type", "partOf"])

            place_id = 0
            country_ids = {}

            # Countries
            for country in COUNTRIES:
                writer.writerow([place_id, country, "Country", ""])
                country_ids[country] = place_id
                place_id += 1

            # Cities
            for country, cities in CITIES.items():
                for city in cities:
                    writer.writerow([place_id, city, "City", country_ids[country]])
                    place_id += 1

        print(f"  Generated {place_id} places")

    def _generate_persons(self):
        """Generate Person nodes with realistic attributes."""
        print("Generating Persons...")
        filepath = self.output_dir / "person.csv"

        base_date = datetime(2020, 1, 1)

        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "firstName", "lastName", "gender", "birthday",
                "creationDate", "locationIP", "browserUsed", "email"
            ])

            for i in tqdm(range(self.sf.num_persons), desc="  Persons"):
                first_name = random.choice(FIRST_NAMES)
                self.first_names_used.append(first_name)

                if HAS_FAKER:
                    last_name = fake.last_name()
                    birthday = fake.date_of_birth(minimum_age=18, maximum_age=70).isoformat()
                    ip = fake.ipv4()
                    browser = random.choice(["Firefox", "Chrome", "Safari", "Edge"])
                else:
                    last_name = f"Lastname{i}"
                    birthday = f"{random.randint(1960, 2000)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                    ip = f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
                    browser = random.choice(["Firefox", "Chrome", "Safari", "Edge"])

                creation_date = (base_date + timedelta(days=random.randint(0, 1825))).isoformat()
                email = f"{first_name.lower()}.{last_name.lower()}{i}@example.com"

                self.person_ids.append(i)
                writer.writerow([
                    i, first_name, last_name,
                    random.choice(["male", "female"]),
                    birthday, creation_date, ip, browser, email
                ])

        print(f"  Generated {len(self.person_ids):,} persons")

    def _generate_knows(self):
        """Generate KNOWS relationships using a preferential attachment model."""
        print("Generating KNOWS relationships...")
        filepath = self.output_dir / "person_knows_person.csv"

        base_date = datetime(2020, 6, 1)
        edges_written = 0

        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Person1Id", "Person2Id", "creationDate"])

            for i in tqdm(range(self.sf.num_persons), desc="  Friendships"):
                # Each person connects to avg_friends others
                num_friends = max(1, int(random.gauss(self.sf.avg_friends, 5)))

                # Preferential attachment: more likely to connect to nearby IDs
                possible_friends = list(range(max(0, i - 1000), i)) + \
                                   list(range(i + 1, min(self.sf.num_persons, i + 1000)))

                if not possible_friends:
                    possible_friends = list(range(self.sf.num_persons))
                    possible_friends.remove(i)

                num_friends = min(num_friends, len(possible_friends))
                friends = random.sample(possible_friends, num_friends)

                for friend in friends:
                    creation_date = (base_date + timedelta(days=random.randint(0, 730))).isoformat()
                    writer.writerow([i, friend, creation_date])
                    edges_written += 1

        print(f"  Generated {edges_written:,} KNOWS relationships")

    def _generate_forums(self):
        """Generate Forum nodes."""
        print("Generating Forums...")
        filepath = self.output_dir / "forum.csv"

        base_date = datetime(2020, 1, 1)

        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "title", "creationDate", "moderatorId"])

            for i in tqdm(range(self.sf.num_forums), desc="  Forums"):
                self.forum_ids.append(i)
                if HAS_FAKER:
                    title = fake.sentence(nb_words=4).replace(",", "")
                else:
                    title = f"Forum {i}: Discussion Topic"
                creation_date = (base_date + timedelta(days=random.randint(0, 1000))).isoformat()
                moderator = random.choice(self.person_ids)
                writer.writerow([i, title, creation_date, moderator])

        print(f"  Generated {len(self.forum_ids):,} forums")

        # Generate forum membership
        self._generate_forum_membership()

    def _generate_forum_membership(self):
        """Generate HAS_MEMBER relationships for forums."""
        filepath = self.output_dir / "forum_hasMember_person.csv"
        members_written = 0

        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ForumId", "PersonId", "joinDate"])

            base_date = datetime(2020, 3, 1)

            for forum_id in self.forum_ids:
                # Each forum has 10-100 members
                num_members = random.randint(10, min(100, len(self.person_ids)))
                members = random.sample(self.person_ids, num_members)

                for person_id in members:
                    join_date = (base_date + timedelta(days=random.randint(0, 1500))).isoformat()
                    writer.writerow([forum_id, person_id, join_date])
                    members_written += 1

        print(f"  Generated {members_written:,} forum memberships")

    def _generate_posts(self):
        """Generate Post/Message nodes."""
        print("Generating Posts...")
        post_filepath = self.output_dir / "post.csv"
        creator_filepath = self.output_dir / "post_hasCreator_person.csv"
        container_filepath = self.output_dir / "forum_containerOf_post.csv"
        tag_filepath = self.output_dir / "post_hasTag_tag.csv"
        reply_filepath = self.output_dir / "comment_replyOf_post.csv"

        base_date = datetime(2020, 6, 1)
        post_id = 0

        with open(post_filepath, "w", newline='') as f_post, \
             open(creator_filepath, "w", newline='') as f_creator, \
             open(container_filepath, "w", newline='') as f_container, \
             open(tag_filepath, "w", newline='') as f_tag, \
             open(reply_filepath, "w", newline='') as f_reply:

            w_post = csv.writer(f_post)
            w_creator = csv.writer(f_creator)
            w_container = csv.writer(f_container)
            w_tag = csv.writer(f_tag)
            w_reply = csv.writer(f_reply)

            w_post.writerow(["id", "imageFile", "creationDate", "locationIP", "browserUsed", "language", "content", "length"])
            w_creator.writerow(["PostId", "PersonId"])
            w_container.writerow(["ForumId", "PostId"])
            w_tag.writerow(["PostId", "TagId"])
            w_reply.writerow(["CommentId", "PostId"])

            for forum_id in tqdm(self.forum_ids, desc="  Posts"):
                num_posts = random.randint(5, self.sf.posts_per_forum)

                for _ in range(num_posts):
                    self.message_ids.append(post_id)

                    if HAS_FAKER:
                        content = fake.paragraph()[:500].replace(",", " ")
                    else:
                        content = f"This is post {post_id} content. Discussing interesting topics..."

                    creation_date = (base_date + timedelta(days=random.randint(0, 1500))).isoformat()
                    creator_id = random.choice(self.person_ids)

                    w_post.writerow([
                        post_id,
                        f"image_{post_id}.jpg" if random.random() < 0.1 else "",
                        creation_date,
                        f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                        random.choice(["Firefox", "Chrome", "Safari"]),
                        random.choice(["en", "de", "fr", "es"]),
                        content,
                        len(content)
                    ])

                    w_creator.writerow([post_id, creator_id])
                    w_container.writerow([forum_id, post_id])

                    # Tags (1-3 tags per post)
                    num_tags = random.randint(1, 3)
                    for tag_id in random.sample(self.tag_ids, min(num_tags, len(self.tag_ids))):
                        w_tag.writerow([post_id, tag_id])

                    # Replies (0-5 replies per post)
                    num_replies = random.randint(0, 5)
                    for reply_offset in range(num_replies):
                        reply_id = post_id * 10000 + reply_offset  # Ensure unique IDs
                        w_reply.writerow([reply_id, post_id])

                    post_id += 1

        print(f"  Generated {len(self.message_ids):,} posts")

    def _generate_likes(self):
        """Generate LIKES relationships."""
        print("Generating Likes...")
        filepath = self.output_dir / "person_likes_post.csv"

        base_date = datetime(2021, 1, 1)
        likes_written = 0

        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["PersonId", "PostId", "creationDate"])

            # Sample of posts to add likes
            sample_posts = random.sample(
                self.message_ids,
                min(len(self.message_ids), len(self.message_ids) // 2)
            )

            for post_id in tqdm(sample_posts, desc="  Likes"):
                num_likes = random.randint(0, 20)
                likers = random.sample(self.person_ids, min(num_likes, len(self.person_ids)))

                for person_id in likers:
                    creation_date = (base_date + timedelta(days=random.randint(0, 700))).isoformat()
                    writer.writerow([person_id, post_id, creation_date])
                    likes_written += 1

        print(f"  Generated {likes_written:,} likes")

    def _write_metadata(self):
        """Write metadata file with generation stats."""
        filepath = self.output_dir / "metadata.json"

        import json
        metadata = {
            "scale_factor": self.sf.name,
            "num_persons": len(self.person_ids),
            "num_messages": len(self.message_ids),
            "num_forums": len(self.forum_ids),
            "num_tags": len(self.tag_ids),
            "generated_at": datetime.now().isoformat(),
            "first_names": list(set(self.first_names_used))[:30],
            "tags": self.tags_used[:50],
        }

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate LDBC-SNB data")
    parser.add_argument(
        "--sf", "--scale-factor",
        type=str,
        default="SF1",
        choices=list(SCALE_FACTORS.keys()),
        help="Scale factor (default: SF1)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="benchmarks/ldbc/data",
        help="Output directory"
    )

    args = parser.parse_args()

    generator = LDBCDataGenerator(scale_factor=args.sf, output_dir=args.output)
    generator.generate_all()


if __name__ == "__main__":
    main()
