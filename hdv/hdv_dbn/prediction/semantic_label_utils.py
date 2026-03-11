import yaml

def load_semantic_labels_from_yaml(yaml_path, S, A):
    """
    Loads semantic labels for each (s, a) pair from a semantic_map.yaml file.
    Returns a list of semantic labels in row-major order: [ (0,0), (0,1), ..., (S-1,A-1) ]
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    if "styles" not in data:
        raise KeyError(f"semantic map missing required top-level key: 'styles' ({yaml_path})")
    if "actions_by_style" not in data:
        raise KeyError(f"semantic map missing required top-level key: 'actions_by_style' ({yaml_path})")

    styles = data["styles"]
    actions_by_style = data["actions_by_style"]
    
    labels = []
    for s in range(S):
        s_key = f"s{s}"
        if s_key not in styles:
            raise KeyError(f"Missing style '{s_key}' in styles ({yaml_path})")
        if s_key not in actions_by_style:
            raise KeyError(f"Missing style '{s_key}' in actions_by_style ({yaml_path})")

        style_name = styles[s_key].get("name", s_key)
        actions_dict = actions_by_style[s_key]

        for a in range(A):
            a_key = f"a{a}"
            if a_key not in actions_dict:
                raise KeyError(
                    f"Missing action '{a_key}' under actions_by_style['{s_key}'] ({yaml_path}). "
                    f"Available: {list(actions_dict.keys())}"
                )
            action_name = actions_dict[a_key].get("name", a_key)
            labels.append(f"{style_name}/{action_name}")

    return labels
