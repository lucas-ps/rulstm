import os
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser(description='Breakfast and 50-Salads action label converter')
parser.add_argument('--dataset_path', type=str, required=True,
                    help='Path to the dataset.')
parser.add_argument('--dataset', type=str, choices=['50-Salads', 'Breakfast'], required=True,
                    help='Dataset to be used (50-Salads/Breakfast).')
args = parser.parse_args()

actions = {}
verbs = {}
nouns = {}
action_id = 0
verb_id = 0
noun_id = 0
records = []
df = pd.DataFrame(records)

# python convert_action_labels.py --dataset_path '/media/lucas/Linux SSD/BreakfastII_15fps_qvga_sync' --dataset Breakfast
# Uses .labels files in the same directory as videos
if args.dataset == 'Breakfast':
    label_files = glob(os.path.join(args.dataset_path, '**', '**', '*.labels'))
    # print(args.dataset_path+ '/**/*.labels')

    for file in label_files:
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            video_name = os.path.basename(file).replace('.avi.labels', '')
            for line in lines:
                frames, action = line.split()
                start_frame, end_frame = frames.split('-')
                
                # Split action into verb and noun
                verb_noun = action.split('_')
                if len(verb_noun) > 1:
                    verb, noun = verb_noun
                else:
                    # 'SIL' case, means start/end of video, no specific action.
                    verb, noun = action, 'none'
                
                if verb not in verbs:
                    verbs[verb] = verb_id
                    verb_id += 1
                if noun not in nouns:
                    nouns[noun] = noun_id
                    noun_id += 1
                if action not in actions:
                    actions[action] = {
                        'action_id': action_id,
                        'verb_id': verbs[verb],
                        'noun_id': nouns[noun]
                    }
                    action_id += 1                

                record = {
                    'video_name': video_name,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'action': actions[action]['action_id'],
                    'verb': verbs[verb],
                    'noun': nouns[noun]
                }
                records.append(record)
    df = pd.DataFrame(records)

# Uses 'groundTruth' from 50-Salads' provided i3d features 
# python convert_action_labels.py --dataset_path '/media/lucas/Linux SSD/50-Salads/' --dataset 50-Salads
else:
    label_files = glob(os.path.join(args.dataset_path, 'groundTruth', '*.txt'))
    #print(label_files)
    for file in label_files:
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            video_name = os.path.basename(file).replace('.txt', '')
            action = "action_start"
            start_frame = 0
            for i, line in enumerate(lines):
                # Check if action has changed
                if line != action:
                    # Save previous action
                    if action != "action_start":
                        end_frame = i - 1
                        # Split into verb and noun
                        verb_noun = action.split('_')
                        if len(verb_noun) > 1:
                            verb, noun = verb_noun[0], "_".join(verb_noun[1:])
                        else:
                            # 'action_start' case, means start/end of video, no specific action.
                            verb, noun = action, 'none'

                        if verb not in verbs:
                            verbs[verb] = verb_id
                            verb_id += 1
                        if noun not in nouns:
                            nouns[noun] = noun_id
                            noun_id += 1
                        if action not in actions:
                            actions[action] = {
                                'action_id': action_id,
                                'verb_id': verbs[verb],
                                'noun_id': nouns[noun]
                            }
                        action_id += 1   

                        record = {
                            'video_name': video_name,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'action': actions[action]['action_id'],
                            'verb': verbs[verb],
                            'noun': nouns[noun]
                        }
                        records.append(record)

                    action = line
                    start_frame = i
    df = pd.DataFrame(records)

action_records = [
    {'action_id': IDs['action_id'], 
    'verb_noun': f"{IDs['verb_id']}_{IDs['noun_id']}", 
    'action': action} 
    for action, IDs in actions.items()
]

# Split the data into training and validation sets
train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)
test_seen_df = pd.concat([train_df, val_df], ignore_index=True)
test_seen_df = test_seen_df.drop(columns=['action', 'verb', 'noun'])
test_df = test_df.drop(columns=['action', 'verb', 'noun'])

# Save validation labels, training labels, actions, verbs, and nouns as separate CSV files
train_df.to_csv('training.csv', index=True, header=False)
val_df.to_csv('validation.csv', index=True, header=False)
test_seen_df.to_csv('test_seen.csv', index=True, header=False)
test_df.to_csv('test_unseen.csv', index=True, header=False)

pd.DataFrame.from_dict(verbs, orient='index').reset_index().to_csv('verbs.csv', index=False, header=['verb', 'verb_id'])
pd.DataFrame.from_dict(nouns, orient='index').reset_index().to_csv('nouns.csv', index=False, header=['noun', 'noun_id'])

action_df = pd.DataFrame(action_records)
action_df.to_csv('actions.csv', index=False, columns=['action_id', 'verb_noun', 'action'], header=False)
