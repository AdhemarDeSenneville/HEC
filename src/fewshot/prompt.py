
CLASS_PROMPT = "You are given an image of a {class_name}\n{current_text_prompt}"

LETTRE_TEMPLATE = {
    "A": 0,  "B": 1,  "C": 2,  "D": 3,  "E": 4,
    "F": 5,  "G": 6,  "H": 7,  "I": 8,  "J": 9,
    "K": 10, "L": 11, "M": 12, "N": 13, "O": 14,
    "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19,
    "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24
}

CD_PROMPT_TEMPLATE = {
    "oxford_pets": {
        "domain_prompt": "What breed is the animal in this image?",
        "clip_prompt": 'a photo of a {class_name}, a type of pet.',
        "end_prompt": "Answer in one word.",
    },
    "eurosat": {
        "domain_prompt": "What type of remote sensing image does the given image belong to?",
        "clip_prompt": 'a centered satellite photo of {class_name}.',
        "end_prompt": "Answer in one word.",
    },
    "ucf101": {
        "domain_prompt": "What action is the person performing in this video frame?",
        "clip_prompt": 'a photo of a person doing {class_name}.',
        "end_prompt": "",
    },
    "sun397": {
        "domain_prompt": "What scene is shown in this image?",
        "clip_prompt": 'a photo of a {class_name}.',
        "end_prompt": "Answer in one word.",
    },
    "caltech101": {
        "domain_prompt": "What is the main object in this photo?",
        "clip_prompt": 'a photo of a {class_name}.',
        "end_prompt": "",
    },
    "dtd": {
        "domain_prompt": "What texture pattern is visible in this image?",
        "clip_prompt": '{class_name} texture.',
        "end_prompt": "",
    },
    "fgvc": {
        "domain_prompt": "Name the aircraft model shown.",
        "clip_prompt": 'a photo of a {class_name}, a type of aircraft.',
        "end_prompt": "Answer in one word.",
    },
    "food101": {
        "domain_prompt": "What is this dish called?",
        "clip_prompt": 'a photo of {class_name}, a type of food.',
        "end_prompt": "Answer in one word.",
    },
    "oxford_flowers": {
        "domain_prompt": "What is the species of this flower?",
        "clip_prompt": 'a photo of a {class_name}, a type of flower.',
        "end_prompt": "",
    },
    "stanford_cars": {
        "domain_prompt": "Which car model is shown in the image?",
        "clip_prompt": 'a photo of a {class_name}.',
        "end_prompt": "Answer in one word.",
    },
}

MD_PROMPT_TEMPLATE = {
    "cu_birds": {
        "domain_prompt": "What is the species of this bird?",
        # remove numbers and underrscores: 040.Olive_sided_Flycatcher -> Olive sided Flycatcher
        "format_class": lambda class_name: " ".join(class_name.split(".")[1].split("_")),
        "end_prompt": "Answer in one word.",
    },
    "traffic_sign": {
        "domain_prompt": "What is the type of this traffic sign?",
        "format_class": lambda class_name: class_name.split(".")[1],
        "end_prompt": "Answer in one word.",
    },
}


def get_prompt_lettre(
        label_to_str, 
        dataset_name, 
        benchmark_name, 
        skip_classes=False,
    ):
    asked_token_type = "class"

    if benchmark_name == "MD":
        prompt_start = MD_PROMPT_TEMPLATE[dataset_name]["domain_prompt"]
        format_class = MD_PROMPT_TEMPLATE[dataset_name]["format_class"]
        prompt_end   = MD_PROMPT_TEMPLATE[dataset_name]["end_prompt"]
    if benchmark_name == "CD":
        prompt_start = CD_PROMPT_TEMPLATE[dataset_name]["domain_prompt"]
        format_class = lambda x: x
        prompt_end   = CD_PROMPT_TEMPLATE[dataset_name]["end_prompt"]
    
    text = prompt_start + "\n"

    if skip_classes:
        if asked_token_type == "lettre":
            return text, {class_name: class_name for class_name in label_to_str.values()}
        elif asked_token_type == "class" or asked_token_type == "sentence":
            return text + prompt_end, {class_name: class_name for class_name in label_to_str.values()}
    
    current_class_to_lettre = {}
    for (class_idx, class_str), lettre in zip(label_to_str.items(), LETTRE_TEMPLATE.keys()):
        class_str_clean = format_class(class_str)

        if asked_token_type == "lettre":
            text += f'{lettre}. {class_str_clean}\n'
            current_class_to_lettre[class_str] = lettre

        elif asked_token_type == "class":
            text += f'{class_str_clean}\n'
            current_class_to_lettre[class_str] = class_str_clean

        elif asked_token_type == "sentence":
            current_class_to_lettre[class_str] = class_str_clean
            if class_idx == 0:
                text += f'Choose between: {class_str_clean}, '
            elif class_idx == len(label_to_str) - 1:
                text += f'and {class_str_clean}.\n'
            else:
                text += f'{class_str_clean}, '
    
    text += prompt_end
    return text, current_class_to_lettre


def decode_prompt_lettre(output_text):

    # output is a list of strings
    output_labels = [
        LETTRE_TEMPLATE.get(lettre_pred, 99999) for lettre_pred in output_text
    ]

    return output_labels


def get_prompt_class(
        label_to_str, 
        dataset_name, 
        benchmark_name,
        current_text_prompt=None,
    ):

    class_prompts = []
    class_prompt_template = CLASS_PROMPT

    if benchmark_name == "MD":
        format_class = MD_PROMPT_TEMPLATE[dataset_name]["format_class"]
    if benchmark_name == "CD":
        format_class = lambda x: x # Already formated
    
    for class_name in label_to_str.values():
        class_name = format_class(class_name)

        class_prompts.append(
            class_prompt_template.format(
                class_name=class_name, 
                current_text_prompt=current_text_prompt,
            )
        ) 
    return class_prompts


def get_prompt_clip(label_to_str, dataset_name, benchmark_name):

    if benchmark_name == "MD":
        format_class = MD_PROMPT_TEMPLATE[dataset_name]["format_class"]

        text = {}
        for class_idx, class_str in label_to_str.items():
            class_str = format_class(class_str)
            text[class_idx] = "This is a photo of a " + class_str

        return text
    

    if benchmark_name == "CD":
        prompt_template = CD_PROMPT_TEMPLATE[dataset_name]["clip_prompt"]

        text = {}
        for class_idx, class_str in label_to_str.items():
            text[class_idx] = prompt_template.format(class_name=class_str)
        
        return text

