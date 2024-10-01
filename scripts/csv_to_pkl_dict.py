import sys 
import csv
import pickle

def main():
    input_files = sys.argv[1:]
    des_dict = {}
    for input_file in input_files:
        with open(input_file) as f:
            for line in csv.reader(f, delimiter=','):
                id = line[0]
                name = line[1]
                description = line[2]
                des_dict[f'http://www.wikidata.org/entity/{id}'] = f'{name}, {description}'
                
    with open('output.pkl', 'wb') as f:
        pickle.dump(des_dict, f)

if __name__ == '__main__':
    main()
