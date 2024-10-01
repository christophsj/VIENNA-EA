import requests
import sys
import csv


def fetch_names_and_descriptions(entity_ids, language='en', batch_size=500):
    def fetch_batch(batch):
        print(f"Fetching data for {len(batch)} entities")
        entity_filter = ' '.join([f"wd:{entity}" for entity in batch])
        query = f"""
            SELECT ?entity ?label ?description
            WHERE {{
            VALUES ?entity {{ {entity_filter} }}
            
            OPTIONAL {{
                ?entity rdfs:label ?label .
                FILTER (LANG(?label) = "{language}" || LANG(?label) = "")
            }}
            
            OPTIONAL {{
                ?entity schema:description ?description .
                FILTER (LANG(?description) = "{language}" || LANG(?description) = "")
            }}
            }}
        """
        
        headers = {}
        url = 'https://query.wikidata.org/sparql'
        response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            batch_results = {}
            for result in data['results']['bindings']:
                entity = result['entity']['value'].split('/')[-1] 
                label = result.get('label', {}).get('value', 'No label available')
                description = result.get('description', {}).get('value', 'No description available')
                batch_results[entity] = {'label': label, 'description': description}
            return batch_results
        else:
            print(f"Error fetching data: {response.status_code}")
            return {}
    
    # Split the list of entity_ids into batches and fetch each batch
    all_results = {}
    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i:i + batch_size]
        batch_results = fetch_batch(batch)
        all_results.update(batch_results)
    
    return all_results
    
def parse_input_ids():
    input_file = sys.argv[1]
    entity_ids = set()
    with open(input_file) as f:
        for line in csv.reader(f, delimiter='\t'):
            current_entities = [line[0], line[2]]
            for entity in current_entities:
                entity_ids.add(entity.split('/')[-1])
        
    
    return list(entity_ids)
    
def main():
    entity_ids = parse_input_ids()
        

    results = fetch_names_and_descriptions(entity_ids)
            
    with open('output.csv', 'w') as f:
        writer = csv.writer(f)
        for entity_id in entity_ids:
            if entity_id in results:
                label = results[entity_id]['label']
                description = results[entity_id]['description']
                writer.writerow([entity_id, label, description])
            else:
                writer.writerow([entity_id, '', ''])

if __name__ == '__main__':
    main()