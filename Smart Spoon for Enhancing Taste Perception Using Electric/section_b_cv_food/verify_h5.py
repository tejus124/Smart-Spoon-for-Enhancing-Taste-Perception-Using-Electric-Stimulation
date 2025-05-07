# # verify_h5.py
# with open('section_b_cv_food/food_model.h5','rb') as f:
#     header = f.read(8)
# print(header)  # should be: b'\x89HDF\r\n\x1a\n'

# verify_food_model.py

with open('section_b_cv_food/food_model.h5', 'rb') as f:
    header = f.read(8)
# print("HDF5 signature:", header)
print("HDF5 signature: b'\x89HDF\r\n\x1a\n'",header)

