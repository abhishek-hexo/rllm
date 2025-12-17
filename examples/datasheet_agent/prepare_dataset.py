from rllm.data.dataset import DatasetRegistry
import os 
import json 


def read_block_diagram_qn(block_diagram_path):
    questions_list = []

    with open(block_diagram_path, 'r') as f:
        block_diagram = json.load(f)
    
    for question_item in block_diagram:
        question = question_item['question']
        answer_key = list(question_item['gold_answer'].keys())[0] 
        answer = {} 
        answer['blocks'] = question_item['gold_answer'][answer_key]["blocks"]
        answer['pins'] = question_item['gold_answer'][answer_key]["pins"]
        answer['description'] = question_item['gold_answer'][answer_key]["description"]
        questions_list.append({
            'question': question,
            'answer': answer, 
            'type': 'block_diagram'
        })

    return questions_list

def read_pin_map_qn(pin_map_path):
    questions_list = [] 

    with open(pin_map_path, 'r') as f:
        pin_map = json.load(f)
    
    for question_item in pin_map:
        question = question_item['question']
        answer_key = list(question_item['gold_answer'].keys())[0] 
        answer = {} 
        answer['pins'] = question_item['gold_answer'][answer_key]['pins']
        answer['description'] = question_item['gold_answer'][answer_key]['description']
        questions_list.append({
            'question': question,
            'answer': answer,
            'type': 'pin_map'
        })
    return questions_list


def read_register_map_qn(register_map_path):
    questions_list = [] 
    with open(register_map_path, 'r') as f:
        register_map = json.load(f)
    
    for question_item in register_map:
        question = question_item['question']
        answer_key = list(question_item['gold_answer'].keys())[0] 
        answer = {} 
        answer['register_map'] = question_item['gold_answer'][answer_key]
        questions_list.append({
            'question': question,
            'answer': answer,
            'type': 'register_map'
        })
    return questions_list

def prepare_dataset():
    def process_split(split='train'):
        processed_data = []
        split_path = f"/lustre/scratch/users/abhishek.maiti/rag/data/{split}.txt"
        with open(split_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            line = line.replace('extracted_data_v2', 'extracted_data_v2_with_PDFs')
            datasheet_id = line.split('/')[-1]
            block_diagram = os.path.join(line, 'block_diagram.json') 
            if os.path.exists(block_diagram):
                questions_list = read_block_diagram_qn(block_diagram)
                for question in questions_list:
                    processed_data.append({
                        'question': question['question'],
                        'ground_truth': question['answer'],
                        'datasheet_id': datasheet_id,
                        'data_source': question['type'],
                    })

            pin_map = os.path.join(line, 'pin_map.json')
            if os.path.exists(pin_map):
                questions_list = read_pin_map_qn(pin_map)
                for question in questions_list:
                    processed_data.append({
                        'question': question['question'],
                        'ground_truth': question['answer'],
                        'datasheet_id': datasheet_id,
                        'data_source': question['type'],
                    })
            register_map = os.path.join(line, 'register_map.json')
            if os.path.exists(register_map):
                questions_list = read_register_map_qn(register_map)
                for question in questions_list:
                    processed_data.append({
                        'question': question['question'],
                        'ground_truth': question['answer'],
                        'datasheet_id': datasheet_id,
                        'data_source': question['type'],
                    })
        return processed_data



    train_data = process_split('train')
    test_data = process_split('test')

    DatasetRegistry.register_dataset('datasheet_agent', train_data, 'train')
    DatasetRegistry.register_dataset('datasheet_agent', test_data, 'test')
    return train_data, test_data

if __name__ == '__main__':
    train_data, test_data = prepare_dataset()
    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")