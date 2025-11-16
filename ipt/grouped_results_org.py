import numpy as np
from collections import defaultdict
from rf100vl.util import get_basename, get_category

# Dictionary of mAP values for different datasets
res = {
    'mAP_actions-zzid2-zb1hq-fsod-amih': 24.064278521404297,
    'mAP_trail-camera-fsod-egos': 56.7836977264109,
    'mAP_paper-parts-fsod-rmrg': 37.17713035390103,
    'mAP_lacrosse-object-detection-fsod-uxkt': 18.044207431906667,
    'mAP_flir-camera-objects-fsod-tdqp': 17.605628602665618,
    'mAP_the-dreidel-project-anzyr-fsod-zejm': 37.85522417418126,
    'mAP_water-meter-jbktv-7vz5k-fsod-ftoz': 26.56982401633573,
    'mAP_orionproducts-vtl2z-fsod-puhv': 25.455224670274234,
    'mAP_aerial-airport-7ap9o-fsod-ddgc': 31.34755646407234,
    'mAP_wildfire-smoke-fsod-myxt': 34.408241015044446,
    'mAP_soda-bottles-fsod-haga': 15.953955247726043,
    'mAP_all-elements-fsod-mebv': 25.076353709933763,
    'mAP_dentalai-i4clz-fsod-fsuo': 9.094968428106615,
    'mAP_wb-prova-stqnm-fsod-rbvg': 51.5895982439329,
    'mAP_aquarium-combined-fsod-gjvb': 32.592390961689794,
    'mAP_x-ray-id-zfisb-fsod-dyjv': 38.44694385039646,
    'mAP_defect-detection-yjplx-fxobh-fsod-amdi': 46.201750742340195,
    'mAP_new-defects-in-wood-uewd1-fsod-tffp': 25.96925844322867,
    'mAP_gwhd2021-fsod-atsv': 16.79637417893668,
    'mAP_recode-waste-czvmg-fsod-yxsw': 32.2488722804605,
    'mAP': 30.164073953147415
}

# Remove the "mAP_" prefix from all keys
res = {k.replace("mAP_", ""): v for k, v in res.items()}

# Extract the overall mAP value
mAP = res.pop("mAP")

# Group datasets by category and collect their mAP values
finals = defaultdict(list)
for key, value in res.items():
    cat = get_category(get_basename(key))
    finals[cat].append(value)

# Print the mean mAP for each category
print("Total mAP: ", mAP)
for k, v in finals.items():
    print(f"{k}: {np.mean(v)}")