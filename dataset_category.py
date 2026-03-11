
import os
from rf100vl.util import get_basename, get_category

DATA_DIR = './datasets/rf100-vl-fsod'


# RF-20 datasets
# {
#     'Industrial': ['recode-waste', 'defect-detection', 'water-meter'], 
#     'Document': ['paper-parts', 'all-elements'], 
#     'Sport': ['lacrosse-object-detection', 'actions'], 
#     'Flora/Fauna': ['trail-camera', 'gwhd2021', 'wb-prova', 'aquarium-combined'], 
#     'Misc': ['orionproducts', 'the-dreidel-project', 'soda-bottles', 'flir-camera-objects', 'new-defects-in-wood'], 
#     'Aerial': ['wildfire-smoke', 'aerial-airport'], 
#     'Lab Imaging': ['dentalai', 'x-ray-id']
# }


# RF-7 datasets for few-shot ablations
# {
#     'Industrial': ['defect-detection'], 
#     'Document': ['all-elements'], 
#     'Sport': ['actions'], 
#     'Flora/Fauna': ['wb-prova'], 
#     'Misc': ['new-defects-in-wood'], 
#     'Aerial': ['aerial-airport'], 
#     'Lab Imaging': ['dentalai']
# }

cat_data_dict = {}
for dataset_name in os.listdir(DATA_DIR):
    cat = get_category(dataset_name)
    print(f"{cat}: {dataset_name}")

    cat_data_dict[cat] = cat_data_dict.get(cat, []) + [dataset_name]

print(cat_data_dict)
