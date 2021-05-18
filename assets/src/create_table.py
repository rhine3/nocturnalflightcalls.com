import pandas as pd
from pathlib import Path

hidden_tds = {
    'alpha_code':"Alpha code",
    'order':"Order",
    'family':"Family",
#    'genus':"Genus",
    'bioacoustic_category':"Bioacoustic category",
    'high_or_low':"Frequency range"
}


# Allow users to toggle columns on/off
start_str = """
<h5>Select columns to display</h5>
<div class="btn-group btn-group-toggle" data-toggle="buttons">
    <label class="btn btn-primary active" id="species_button">
        <input type="checkbox" name="options" autocomplete="off" checked onchange="toggleHiddenColumn('species', 'species_button')"> Species
    </label>"""
# Selectors for the hidden columns
for col_class, col_title in hidden_tds.items():
    start_str += f"""
    <label class="btn btn-primary" id="{col_class}_button">
        <input type="checkbox" name="options" autocomplete="off" onchange="toggleHiddenColumn('{col_class}', '{col_class}_button')"> {col_title}
    </label>"""
start_str += """
    <label class="btn btn-primary active" id="spectrograms_button">
        <input type="checkbox" name="options" autocomplete="off" checked onchange="toggleHiddenColumn('spectrograms', 'spectrograms_button')"> Spectrograms
    </label>
</div><br><br>"""


# Header of the table with hidden columns hidden by default
start_str +="""
<h5>Click on a row to display more spectrogram examples</h5>
<table>
  <thead>
      <tr>
        <th class="species">Species</th>"""
for col_class, col_title in hidden_tds.items():
    start_str += f"""
        <th class="{col_class}" style="display:none;">{col_title}</th>"""
start_str += """
        <th class="spectrograms" width="60%">Spectrograms</th>
      </tr>
  </thead>
  <tbody id="nfcTable">"""

top_dir = Path("/Users/tessa/Code/nfcs2/")
df = pd.read_csv(top_dir.joinpath("assets/src/table_source_realistic_example.csv"))

for idx, row in df.iterrows():
    common_name = row['common_name']
    scientific_name = row['scientific_name']
    start_str += f"""
    <tr onclick="toggleHiddenImage('image_toggle{idx}');" class="species_row">
        <td class="species">{common_name} (<i>{scientific_name}</i>)</td>"""

    # Get list of other searchable categories that won't display (as of yet)
    hidden_col_classes = hidden_tds.keys()
    for col_class, data in zip(hidden_col_classes, row[hidden_col_classes]):
        if data == 'nan':
            print(obj)
            data = ''
        start_str += f"""
        <td class={col_class} style="display:none;">{data}</td>"""

    # Get list of images
    alpha_code = row['alpha_code']
    image_loc = 'assets/spectrograms/'+alpha_code
    image_urls = list(top_dir.joinpath(image_loc).glob("*.png"))
    img_str = ''
    images_to_show = 5
    for image_url in image_urls[:images_to_show]:
        img_str += f'<img src="{top_dir.joinpath(image_url)}" width="150" height="100"> '
    for image_url in image_urls[images_to_show:]:
        img_str += f'<img src="{top_dir.joinpath(image_url)}" width="150" height="100" class="image_toggle{idx}" style="display:none"> '

    start_str += f"""
        <td class="spectrograms">
            {img_str}
        </td>
    </tr>"""

start_str += """
    </tbody>
</table>
"""

print(start_str)
with open(top_dir.joinpath("assets/src/table.html"), "w+") as f:
    f.write(start_str)
