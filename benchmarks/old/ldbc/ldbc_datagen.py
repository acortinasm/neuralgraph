import csv
import random
import os
import datetime
from faker import Faker
from tqdm import tqdm

fake = Faker()
Faker.seed(42)
random.seed(42)

SCALE_FACTOR = 0.1  # Target ~1000 nodes for initial test
NUM_PERSONS = int(1000 * SCALE_FACTOR * 10) # 1000 persons
NUM_FORUMS = int(NUM_PERSONS / 10)
NUM_POSTS = NUM_PERSONS * 5

DATA_DIR = "benchmarks/ldbc/data"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_persons():
    print("Generating Persons...")
    with open(f"{DATA_DIR}/person.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "firstName", "lastName", "gender", "birthday", "creationDate", "locationIP", "browserUsed"])
        
        for i in range(NUM_PERSONS):
            writer.writerow([
                i,
                fake.first_name(),
                fake.last_name(),
                random.choice(["male", "female"]),
                fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
                fake.date_time_this_decade().isoformat(),
                fake.ipv4(),
                fake.user_agent().replace(",", "") # Simple CSV safety
            ])

def generate_knows():
    print("Generating Knows (Friendships)...")
    with open(f"{DATA_DIR}/person_knows_person.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Person1Id", "Person2Id", "creationDate"])
        
        for i in range(NUM_PERSONS):
            # Each person knows ~10-20 others (Small World)
            num_friends = random.randint(5, 20)
            friends = random.sample(range(NUM_PERSONS), num_friends)
            for friend in friends:
                if i == friend: continue
                # Undirected in theory, but directed in CSV. nGraph treats as directed usually.
                writer.writerow([i, friend, fake.date_time_this_year().isoformat()])

def generate_forums_and_posts():
    print("Generating Forums and Posts...")
    
    with open(f"{DATA_DIR}/forum.csv", "w") as f_forum, \
         open(f"{DATA_DIR}/post.csv", "w") as f_post, \
         open(f"{DATA_DIR}/forum_containerOf_post.csv", "w") as f_cont, \
         open(f"{DATA_DIR}/post_hasCreator_person.csv", "w") as f_creator:
        
        w_forum = csv.writer(f_forum)
        w_post = csv.writer(f_post)
        w_cont = csv.writer(f_cont)
        w_creator = csv.writer(f_creator)
        
        w_forum.writerow(["id", "title", "creationDate"])
        w_post.writerow(["id", "imageFile", "creationDate", "locationIP", "browserUsed", "language", "content", "length"])
        w_cont.writerow(["ForumId", "PostId"])
        w_creator.writerow(["PostId", "PersonId"])
        
        post_id_counter = 0
        
        for i in range(NUM_FORUMS):
            forum_date = fake.date_time_this_decade()
            w_forum.writerow([i, fake.sentence(nb_words=4), forum_date.isoformat()])
            
            # Posts in forum
            num_posts_in_forum = random.randint(5, 50)
            for _ in range(num_posts_in_forum):
                pid = post_id_counter
                post_id_counter += 1
                
                # Creator
                creator_id = random.randint(0, NUM_PERSONS - 1)
                
                content = fake.paragraph()
                w_post.writerow([
                    pid,
                    f"image_{pid}.jpg" if random.random() < 0.1 else "",
                    fake.date_time_between(start_date=forum_date).isoformat(),
                    fake.ipv4(),
                    fake.user_agent().replace(",", ""),
                    fake.language_code(),
                    content.replace("\n", " ").replace(",", ""),
                    len(content)
                ])
                
                w_cont.writerow([i, pid])
                w_creator.writerow([pid, creator_id])

def main():
    generate_persons()
    generate_knows()
    generate_forums_and_posts()
    print(f"Data generated in {DATA_DIR}")

if __name__ == "__main__":
    main()
