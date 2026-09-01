import cv2
import numpy as np

img = np.ones((200, 200, 3), dtype=np.uint8) * 255
cv2.circle(img, (100, 100), 60, (0, 0, 0), -1)
cv2.circle(img, (100, 100), 30, (255, 255, 255), -1)

edges = cv2.Canny(img, 20, 60)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]

nodes = {}
for i, c in enumerate(contours):
    level = 0
    parent = hierarchy[i][3]
    while parent != -1:
        level += 1
        parent = hierarchy[parent][3]
    nodes[i] = {'valid': True, 'parent': hierarchy[i][3], 'level': level, 'children': []}

for i, node in nodes.items():
    if node['level'] % 2 != 0:
        node['valid'] = False
    else:
        node['level'] = node['level'] // 2
        node['is_hole'] = (node['level'] % 2 != 0)

for i, node in nodes.items():
    if not node['valid']: continue
    p = node['parent']
    while p != -1 and not nodes[p]['valid']:
        p = nodes[p]['parent']
    node['parent'] = p
    if p != -1:
        nodes[p]['children'].append(i)

for i, node in nodes.items():
    if node['valid']:
        print(f"Valid Contour {i}: parent={node['parent']}, level={node['level']}, is_hole={node['is_hole']}")
