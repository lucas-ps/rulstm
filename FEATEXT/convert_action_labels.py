verb_dict = {}
noun_dict = {}
action_dict = {}
verb_id = 0
noun_id = 0
action_id = 0

with open("original_annotations.txt", "r") as file:
    for line in file:
        _, action = line.split()
        # Assumes (mostly correctly) that action labels are two words, verb first then noun. Eg. pour_cereals
        verb, noun = action.split('_')

        if verb not in verb_dict:
            verb_dict[verb] = verb_id
            verb_id += 1

        if noun not in noun_dict:
            noun_dict[noun] = noun_id
            noun_id += 1

        if action not in action_dict:
            action_dict[action] = action_id
            action_id += 1