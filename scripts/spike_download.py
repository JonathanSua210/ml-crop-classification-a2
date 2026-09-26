"""Throwaway: verify the dataset downloads and inspect one sample."""
import breizhcrops as bzh

dataset = bzh.BreizhCrops("frh04")

print(f"parcels: {len(dataset)}")

x, y, field_id = dataset[0]
print(f"x shape: {x.shape}  dtype: {x.dtype}")
print(f"y: {y}  field_id: {field_id}")
print(f"classes: {dataset.classname}")
