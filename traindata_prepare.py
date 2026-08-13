import json
import csv
import re
import os


def extract_messages_from_json(input_file, output_txt_file):
    """
    Reads a JSON file, extracts the 'message' field from every message
    object, and saves them into a text file (one message per line).
    Specifically for the twitch-download pipeline setup.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract only the 'message' strings
        messages = [
            msg['message']
            for msg in data.get('messages', [])
            if 'message' in msg
        ]

        with open(output_txt_file, 'w', encoding='utf-8') as f:
            for msg in messages:
                f.write(msg + '\n')

        print(
            f"Successfully extracted {len(messages)} "
            f"messages to {output_txt_file}"
        )

    except Exception as e:
        print(f"Error extracting messages: {e}")


def transform_messages_to_vectors(
    input_txt_file,
    lookup_csv_file,
    output_json_file = "default.json",
    return_dataset="false"):
    """
    Takes a text file of messages and a CSV lookup table.

    The CSV should have two columns:
        word,integer

    Each message is transformed into a list of integers based on
    the lookup table.

    Example:
        "this is great"
        -> [5, 20, 100]

    The final JSON contains one vector per message.
    """

    lookup_table = {}
    
    # ------------------------------------------
    # 1. Load the lookup table
    # ------------------------------------------
    try:
        with open(lookup_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) != 2:
                    continue

                word, val = row

                lookup_table[word.lower().strip()] = int(val)

    except Exception as e:
        print(f"Error reading lookup table: {e}")
        return

    # ------------------------------------------
    # 2. Transform messages
    # ------------------------------------------
    dataset = []

    try:
        with open(input_txt_file, 'r', encoding='utf-8') as f:
        
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # Convert to lowercase and split into words.
                #
                # Example:
                # "This is Great!"
                # -> ["this", "is", "great"]
                words = re.findall(r'\w+', line.lower())

                # Convert each word to its integer ID.
                # Words not found in the lookup table are turned to 0.
                vector = [
                    lookup_table.get(word, 0)
                    for word in words
                ]


                # Only keep messages where at least one word
                # was found in the lookup table.
                if vector:
                    dataset.append(vector)

        # ------------------------------------------
        # 3. Save dataset
        # ------------------------------------------
        
        if return_dataset == "true":
            return dataset
        
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f)

        print(
            f"Successfully transformed {len(dataset)} "
            f"messages into vectors."
        )
        print(f"Saved dataset to {output_json_file}")

    except Exception as e:
        print(f"Error transforming messages: {e}")










def create_streamer_vectors(
    streamer_name,
    input_directory,
    lookup_csv_file,
    output_directory
):
    """
    Processes all JSON chat files belonging to one streamer.

    Each JSON file is:
        JSON chat
        -> extracted messages
        -> integer vectors

    All vectors from all chat files are combined into one JSON file:

        output_directory/streamer_name.json
    """

    os.makedirs(output_directory, exist_ok=True)

    streamer_directory = os.path.join(
        input_directory,
        streamer_name
    )

    output_file = os.path.join(
        output_directory,
        f"{streamer_name}.json"
    )

    all_vectors = []

    # ------------------------------------------
    # Find all JSON files for this streamer
    # ------------------------------------------

    json_files = [
        file
        for file in os.listdir(streamer_directory)
        if file.lower().endswith(".json")
    ]

    json_files.sort()

    print(
        f"Found {len(json_files)} JSON files for "
        f"{streamer_name}"
    )

    # ------------------------------------------
    # Process every JSON file
    # ------------------------------------------

    for i, json_file in enumerate(json_files, start=1):

        input_json = os.path.join(
            streamer_directory,
            json_file
        )

        # Temporary files
        extracted_file = os.path.join(
            output_directory,
            "_temp_messages.txt"
        )

        temp_vectors_file = os.path.join(
            output_directory,
            "_temp_vectors.json"
        )

        print(
            f"[{i}/{len(json_files)}] "
            f"Processing {json_file}"
        )

        # Step 1:
        # JSON -> messages
        extract_messages_from_json(
            input_json,
            extracted_file
        )

        # Step 2:
        # messages -> vectors
        transform_messages_to_vectors(
            extracted_file,
            lookup_csv_file,
            temp_vectors_file
        )

        # --------------------------------------
        # Load the vectors
        # --------------------------------------

        try:
            with open(
                temp_vectors_file,
                "r",
                encoding="utf-8"
            ) as f:
                vectors = json.load(f)

            all_vectors.extend(vectors)

        except Exception as e:
            print(
                f"Error reading vectors from "
                f"{json_file}: {e}"
            )

    # ------------------------------------------
    # Save combined vectors
    # ------------------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            all_vectors,
            f
        )

    print()
    print(
        f"Finished {streamer_name}: "
        f"{len(all_vectors)} messages"
    )

    print(
        f"Saved to {output_file}"
    )

    # ------------------------------------------
    # Clean temporary files
    # ------------------------------------------

    if os.path.exists(extracted_file):
        os.remove(extracted_file)

    if os.path.exists(temp_vectors_file):
        os.remove(temp_vectors_file)
