# thesis_dataset

- train a diffusion model on relations between shapes
- (shape1, shape2, color1, color2, relation)
- ex. (circle, triangle, black, white, circle above)
- give diffusion model the image and tuple, then try to generate images based on new tuples

# tuples for each data category:

**no_overlap**
- (circle, triangle, black, red, separate)

**no_overlap_bw**
- (circle, triangle, black, black, separate)

**overlap_circle**
- (circle, triangle, black, red, circle above)

**overlap_circle_bw**
- (circle, triangle, black, black, circle above)

**overlap_triangle**
- (circle, triangle, black, red, triangle above)

**overlap_triangle_bw**
- (circle, triangle, black, black, triangle above)

# next steps for this week
- train basic models, basic MLP, complex CNN, fine tune image recognition model