import yaml

def load_semantic_labels_from_yaml(yaml_path, S, A):
    """
    Loads semantic labels for each (s, a) pair from a semantic_map.yaml file.
    Returns a list of semantic labels in row-major order: [ (0,0), (0,1), ..., (S-1,A-1) ]
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Build mapping from (s, a) to semantic string: "style_name/action_name"
    styles = data['styles']
    labels = []
    for s in range(S):
        s_key = f's{s}'
        style_name = styles[s_key]['name']
        actions = styles[s_key]
        for a in range(A):
            a_key = f'a{a}'
            action_name = actions[a_key]['name']
            labels.append(f"{style_name}/{action_name}")
    return labels
