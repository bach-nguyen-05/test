from collections import Counter
import os
import base64

def majority_vote(lst):
    lst = [x for x in lst if x != ""]
    if not lst:
        return ""
    counts = Counter(lst)
    most_common = counts.most_common(1)
    return most_common[0][0]

# html visualization
html_head = '''
<!DOCTYPE html>
<html>
<style>
table, th, td {
  border:1px solid white;
}
body{
background-color: black;
color: white;
}
.qa {
    width: 150px;
    position: sticky;
    top: 24px;
    left: 300px;
    background: black;
}
.image {
    position: sticky;
    top: 24px;
    left: 0px;
    background: black;
}
th {
  position: sticky;
  top: 0px;
  max-width: 10%;
  background: black;
}
.prediction {
    max-width: 300px;
    max-height: 500px;
    overflow-y: scroll;
}
</style>
<body>
'''
html_tail = '''
</body>
</html>
'''

# visualize predictions, results is a dictionary with keys as model names and values as list of predictions
def visualize_predictions(results, test_data, output_path, image_dir=""):
    img_th_style = "style=\"width:300px;\""
    th = ["<th>Index</th>", f"<th {img_th_style}>Image</th>", "<th>Q&A</th>"] + [f"<th>{model}</th>" for model in results]
    th = "\n".join(th)
    html_table = [f'''
    <table style="width:100%">
    <tr>
        {th}
    </tr>
    ''']
    for i in range(len(test_data)):
        row = ["<tr>"]
        d = test_data[i]
        q = d["question"]
        a = d["ground_truth"]
        if image_dir:
            img_path = os.path.join(image_dir, d["path"])
        else:
            img_path = d["path"]
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('ascii')
        row.append(f'''<td>{i}</td>''')
        row.append(f'''<td class=\"image\" ><img src="data:image/png;base64, {encoded_string}" {img_th_style} /></td>''')
        row.append(f'''<td class=\"qa\" >Q: {q} <br> A: {a}</td>''')
        for model in results:
            # e = exp_files[exp][i]
            if type(results[model][i]) == list:
                e_a = results[model][i][0]
                tf = results[model][i][1]
                color = "green" if tf else "red"
            else:
                e_a = results[model][i]
                color = "white"
            prediction = e_a.replace("\n", "<br>").replace("<cot>", "<span style=\"color: orange;\">").replace("</cot>", "</span>")
            row.append(f'''<td style="color:{color};"><p class="prediction">{prediction}</p></td>''')
        row.append("</tr>")
        html_table.append("\n".join(row))
    html_table += ['''</table>''']
    html = html_head + "\n".join(html_table) + html_tail
    with open(output_path, "w") as f:
        f.write(html)