import re
import pickle
import csv

def main():
    ent_links = [
        '/home/christoph/TU/12_SS24/MA/PRASE-Python/data/EN_DE_15K_V2/ent_links',
        '/home/christoph/TU/12_SS24/MA/PRASE-Python/data/EN_FR_15K_V2/ent_links',
        '/home/christoph/TU/12_SS24/MA/PRASE-Python/data/D_Y_15K_V2/ent_links'
    ]
    file_path = ['/home/christoph/Downloads/short-abstracts_lang=de.ttl',
                 '/home/christoph/Downloads/short-abstracts_lang=fr.ttl',
                 '/home/christoph/Downloads/short-abstracts_lang=en.ttl']
    all_entity_ids = parse_csv_to_set(ent_links)
    entity_dict = parse_ttl_to_dict(file_path)
    print(f"Number of entities: {len(entity_dict)}")
    print(f"Number of ids: {len(all_entity_ids)}")
    # print some ids of set
    print(list(all_entity_ids)[:10])
    
    # filter entity dict by all_entity_ids
    entity_dict = {k: v for k, v in entity_dict.items() if k in all_entity_ids}
    print(f"Number of entities: {len(entity_dict)}")
    save_dict_to_pickle(entity_dict, 'des_dict.pkl')

def parse_csv_to_set(file_paths):
    # Set to store all unique values from the CSV files
    values_set = set()
    
    # Iterate through each file path
    for file_path in file_paths:
        # Read the CSV file
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            
            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Add each value from the row to the set
                for value in row:
                    # Check if the value is from dppedia
                    if value.startswith('http'):
                        values_set.add(value)
    
    return values_set

def parse_ttl_to_dict(file_paths: list):
    # Dictionary to store the parsed data
    entities_dict = {}
    
    
    # Regular expression to match the pattern in the TTL file
    
    # Read the file and parse each line
    for file_path in file_paths:
        if file_path.endswith('de.ttl'):
            pattern = re.compile(r'<(.*?)> <http://www.w3.org/2000/01/rdf-schema#comment> "(.*?)"@de .')
        elif file_path.endswith('fr.ttl'):
            pattern = re.compile(r'<(.*?)> <http://www.w3.org/2000/01/rdf-schema#comment> "(.*?)"@fr .')
        elif file_path.endswith('en.ttl'):
            pattern = re.compile(r'<(.*?)> <http://www.w3.org/2000/01/rdf-schema#comment> "(.*?)"@en .')
            
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = pattern.match(line)
                if match:
                    entity = match.group(1)
                    description = match.group(2)
                    entities_dict[entity] = description
    
    return entities_dict

def save_dict_to_pickle(dictionary, output_path):
    # Write the dictionary to a pickle file
    with open(output_path, 'wb') as file:
        pickle.dump(dictionary, file)


if __name__ == '__main__':
    main()