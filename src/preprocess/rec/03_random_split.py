import argparse
import os
import collections
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--seq_path', type=str)
    parser.add_argument('--zero', action='store_true', help='if true, will process for 0-core, else for 5-core (by default)')
    return parser.parse_args()

def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), collections.defaultdict(list)
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        new_inters[user] = user_inters
    return new_inters

def split_data(user_inters):
    total_len = len(user_inters)
    train_len = int(total_len * 0.8)
    val_len = int(train_len * 0.1)
    
    # Shuffle and split
    random.shuffle(user_inters)
    train_inters = user_inters[:train_len]
    val_inters = train_inters[:val_len]
    test_inters = user_inters[train_len:]
    train_inters = train_inters[val_len:]
    
    return train_inters, val_inters, test_inters

if __name__ == '__main__':
    args = parse_args()

    if args.zero:
        args.input_path = args.input_path.replace('5core', '0core')
        args.output_path = args.output_path.replace('5core', '0core')
        args.seq_path = args.seq_path.replace('5core', '0core')
        print(args)

    all_files = os.listdir(args.input_path)
    for single_file in all_files:
        assert single_file.endswith('.csv')
        prefix = single_file[:-len('.csv')]
        args.file_path = os.path.join(args.input_path, single_file)
        print(args.file_path)

        inters = []
        with open(args.file_path, 'r') as file:
            for line in tqdm(file, 'Loading'):
                user_id, item_id, rating, timestamp = line.strip().split(',')
                timestamp = int(timestamp)
                inters.append((user_id, item_id, rating, timestamp))

        ordered_inters = make_inters_in_order(inters=inters)

        # For direct recommendation
        train_file = open(os.path.join(args.output_path, f'{prefix}.train.csv'), 'w')
        valid_file = open(os.path.join(args.output_path, f'{prefix}.valid.csv'), 'w')
        test_file = open(os.path.join(args.output_path, f'{prefix}.test.csv'), 'w')

        for user in tqdm(ordered_inters, desc='Write direct files'):
            train_inters, val_inters, test_inters = split_data(ordered_inters[user])
            
            for inter in train_inters:
                train_file.write(f'{inter[0]},{inter[1]},{inter[2]},{inter[3]}\n')
            
            for inter in val_inters:
                valid_file.write(f'{inter[0]},{inter[1]},{inter[2]},{inter[3]}\n')
            
            for inter in test_inters:
                test_file.write(f'{inter[0]},{inter[1]},{inter[2]},{inter[3]}\n')

        for file in [train_file, valid_file, test_file]:
            file.close()

        # For sequential recommendation
        train_file = open(os.path.join(args.seq_path, f'{prefix}.train.csv'), 'w')
        valid_file = open(os.path.join(args.seq_path, f'{prefix}.valid.csv'), 'w')
        test_file = open(os.path.join(args.seq_path, f'{prefix}.test.csv'), 'w')

        for user in tqdm(ordered_inters, desc='Write seq files'):
            train_inters, val_inters, test_inters = split_data(ordered_inters[user])
            
            for inter in train_inters:
                history = ' '.join([i[1] for i in train_inters if i[3] < inter[3]])
                train_file.write(f'{inter[0]},{inter[1]},{inter[2]},{inter[3]},{history}\n')
            
            for inter in val_inters:
                history = ' '.join([i[1] for i in train_inters + val_inters if i[3] < inter[3]])
                valid_file.write(f'{inter[0]},{inter[1]},{inter[2]},{inter[3]},{history}\n')
            
            for inter in test_inters:
                history = ' '.join([i[1] for i in train_inters + val_inters + test_inters if i[3] < inter[3]])
                test_file.write(f'{inter[0]},{inter[1]},{inter[2]},{inter[3]},{history}\n')

        for file in [train_file, valid_file, test_file]:
            file.close()
