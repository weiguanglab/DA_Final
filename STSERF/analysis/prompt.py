from api import call_deekseek_api


def create_prompt(request: str, shopping_list: list, k=10) -> str:
    numbered_items = []
    for item in shopping_list:
        numbered_items.append(f"item_id:{item.id}. item_metadata:{item.metadata}")

    items_text = "\n\t".join(numbered_items)

    prompt = f"""Please sort the shopping list based on the
following requirement:

User requirement: {request}

Shopping list:
\t{items_text}

Please sort the above shopping list item's item_id from most suitable to least suitable according to how well their metadata match the user requirement "{request}".

Sorting requirements:
1. Carefully analyze the matching degree between each item's metadata and the requirement
2. Sort from highest to lowest matching degree
3. Provide a brief explanation for each item's ranking
4. Output format: Rank. item_id - Reason

Here are some examples of the desired output format:

Example 1:
If the user requirement is "a red t-shirt for sports" and the shopping list contains:
    item_id:TS001. item_metadata:Red sports t-shirt, breathable fabric, quick-dry.
    item_id:TS002. item_metadata:Blue casual t-shirt, cotton.
    item_id:TS003. item_metadata:Red formal shirt, silk.

The output should be:
1. TS001 - This is a red t-shirt specifically designed for sports, matching all criteria.
2. TS003 - This shirt is red, which matches one criterion, but it's a formal shirt, not for sports.
3. TS002 - This t-shirt is casual and blue, not matching the color or primary use case.

Example 2:
If the user requirement is "durable and waterproof hiking boots" and the shopping list contains:
    item_id:HB001. item_metadata:Men's hiking boots, GORE-TEX waterproof, Vibram sole, full-grain leather.
    item_id:HB002. item_metadata:Lightweight trail running shoes, breathable mesh, not waterproof.
    item_id:HB003. item_metadata:Fashion ankle boots, suede, not for hiking.

The output should be:
1. HB001 - These boots are explicitly waterproof, made of durable leather, and designed for hiking.
2. HB002 - These are for trail running and not waterproof, less suitable for the hiking boot requirement.
3. HB003 - These are fashion boots and not suitable for hiking or waterproof needs.

Please start sorting:"""

    return prompt


def extract_list_from_response(response: str, shopping_list: list) -> list:
    import re

    original_items = set([item.id for item in shopping_list])
    sorted_list = []

    lines = response.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        pattern = r'^\d+\.\s*([^-]+)\s*-'
        match = re.match(pattern, line)

        if match:
            item_name = match.group(1).strip()
            if item_name in original_items:
                sorted_list.append(item_name)
        else:
            for item in shopping_list:
                if item in line and item not in sorted_list:
                    sorted_list.append(item)
                    break

    for item in shopping_list:
        if item not in sorted_list:
            sorted_list.append(item)

    return sorted_list
