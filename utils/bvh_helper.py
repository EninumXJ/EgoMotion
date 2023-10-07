import utils.bvh2np as BVHLIB
import math
import copy
import numpy as np
from pathlib import Path

class BVHChannel(object):
    ChannelTransformMatrixMap = {
            'Xposition': lambda x: np.array([[1, 0, 0, x],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]]),
            'Yposition': lambda x: np.array([[1, 0, 0, 0],
                                             [0, 1, 0, x],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]]),
            'Zposition': lambda x: np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, x],
                                             [0, 0, 0, 1]]),
            'Xrotation': lambda x: np.array([[1, 0, 0, 0],
                                             [0, math.cos(math.radians(x)), -math.sin(math.radians(x)), 0],
                                             [0, math.sin(math.radians(x)), math.cos(math.radians(x)), 0],
                                             [0, 0, 0, 1]]),
            'Yrotation': lambda x: np.array([[math.cos(math.radians(x)), 0, math.sin(math.radians(x)), 0],
                                             [0, 1, 0, 0],
                                             [-math.sin(math.radians(x)), 0, math.cos(math.radians(x)), 0],
                                             [0, 0, 0, 1]]),
            'Zrotation': lambda x: np.array([[math.cos(math.radians(x)), -math.sin(math.radians(x)), 0, 0],
                                             [math.sin(math.radians(x)), math.cos(math.radians(x)), 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]])
        }
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.value = 0.0

    def set_value(self, value):
        self.value = value

    def matrix(self):
        return BVHChannel.ChannelTransformMatrixMap[self.name](self.value)

    def str(self):
        return 'Channel({name}) = {value}'.format(name=self.name, value=self.value)

class BVHNode(object):
    def __init__(self, name, offsets, channel_names, children, weight=1):
        super().__init__()
        self.name = name
        self.children = children # []
        self.channels = [BVHChannel(cn) for cn in channel_names] # []
        self.offsets = offsets # x, y, z
        # weight for calculate frame-frame distance
        self.weight = weight

    def search_node(self, name):
        if self.name == name:
            return self
        for child in self.children:
            result = child.search_node(name)
            if result:
                return result
        return None

    def __load_frame(self, frame_data_array):
        ''' 
            this function modify frame_data_array, so 
            make sure you only call load_frame instead of this
        '''
        for channel in self.channels:
            channel.set_value(frame_data_array.pop(0))
        for child in self.children:
            child.__load_frame(frame_data_array)

    def load_frame(self, frame_data_array):
        frame_data_array = copy.copy(frame_data_array)
        self.__load_frame(frame_data_array)

    def apply_transformation(self, parent_tran_matrix=np.identity(4)):
        self.coordinates = np.zeros((3,1))
        local_translation = np.array([[1, 0, 0, self.offsets[0]],
                                     [0, 1, 0, self.offsets[1]],
                                     [0, 0, 1, self.offsets[2]],
                                     [0, 0, 0, 1]])
        tran_matrix = np.identity(4)
        tran_matrix = np.dot(tran_matrix, parent_tran_matrix)
        tran_matrix = np.dot(tran_matrix, local_translation)
        for channel in self.channels:
            tran_matrix = np.dot(tran_matrix, channel.matrix())
        self.coordinates = np.dot(tran_matrix, np.append(self.coordinates, [[1]], axis=0))[:3]
        for child in self.children:
            child.apply_transformation(tran_matrix)

    def str(self, show_coordinates=False):
        s = 'Node({name}), offset({offset})\n'\
                .format(name=self.name,
                        offset=', '.join([str(o) for o in self.offsets]))
        if show_coordinates:
            try:
                s = s + '\tWorld coordinates: (%.2f, %.2f, %.2f)\n' % (self.coordinates[0],
                                                                       self.coordinates[1],
                                                                       self.coordinates[2])
            except Exception as e:
                print('World coordinates is not available, call apply_transformation() first')
        s = s + '\tChannels:\n'
        for channel in self.channels:
            s = s + '\t\t' + channel.str() + '\n'
        for child in self.children:
            lines = child.str(show_coordinates=show_coordinates).split('\n')
            for line in lines:
                s = s + '\t' + line + '\n'
        return s

    def distance(node_a, node_b):
        assert(node_a.name == node_b.name and node_a.weight == node_b.weight)
        distance = np.linalg.norm(node_a.coordinates - node_b.coordinates) * node_a.weight
        for child_a, child_b in zip(node_a.children, node_b.children):
            distance += BVHNode.distance(child_a, child_b)
        return distance

    def frame_distance(self, frame_a, frame_b):
        root_a = copy.deepcopy(self)
        root_a.load_frame(frame_a)
        root_a.apply_transformation()
        root_b = copy.deepcopy(self)
        root_b.load_frame(frame_b)
        root_b.apply_transformation()
        return BVHNode.distance(root_a, root_b)


def parse_bvh_node(bvhlib_node):
    '''This function parses object from bvh-python (https://github.com/20tab/bvh-python)'''
    name = bvhlib_node.name
    offsets = [float(f) for f in bvhlib_node.children[0].value[1:]]
    channel_names = []
    for channels in bvhlib_node.filter('CHANNELS'):
        channel_names = [c for c in channels.value[2:]]
    children = []
    for c in bvhlib_node.filter('JOINT'):
        children.append(parse_bvh_node(c))
    node = BVHNode(name, offsets,
                   channel_names, children)
    return node

def loads(s):
    bvhlib = BVHLIB.Bvh(s)
    root = parse_bvh_node(bvhlib.get_joints()[0])
    return root, [[float(f) for f in frame] for frame in bvhlib.frames], bvhlib.frame_time

def load(file_path):
    with open(file_path, 'r') as f:
        return loads(f.read())
        

class BvhNode(object):
    def __init__(
        self, name, offset, rotation_order,
        children=None, parent=None, is_root=False, is_end_site=False):
        if not is_end_site and \
          rotation_order not in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
            raise ValueError(f'Rotation order invalid.')
        self.name = name
        self.offset = offset
        self.rotation_order = rotation_order
        self.children = children
        self.parent = parent
        self.is_root = is_root
        self.is_end_site = is_end_site
    

class BvhHeader(object):
    def __init__(self, root, nodes):
        self.root = root
        self.nodes = nodes


def write_header(writer, node, level):
    indent = ' ' * 4 * level
    if node.is_root:
        writer.write(f'{indent}ROOT {node.name}\n')
        channel_num = 6
    elif node.is_end_site:
        writer.write(f'{indent}End Site\n')
        channel_num = 0
    else:
        writer.write(f'{indent}JOINT {node.name}\n')
        channel_num = 3
    writer.write(f'{indent}{"{"}\n')

    indent = ' ' * 4 * (level + 1)
    writer.write(
        f'{indent}OFFSET '
        f'{node.offset[0]} {node.offset[1]} {node.offset[2]}\n'
    )
    if channel_num:
        channel_line = f'{indent}CHANNELS {channel_num} '
        if node.is_root:
            channel_line += f'Xposition Yposition Zposition '
        channel_line += ' '.join([
            f'{axis.upper()}rotation'
            for axis in node.rotation_order
        ])
        writer.write(channel_line + '\n')
    
    for child in node.children:
        write_header(writer, child, level + 1)
    
    indent = ' ' * 4 * level
    writer.write(f'{indent}{"}"}\n')


def write_bvh(output_file, header, channels, frame_rate=30):
    output_file = Path(output_file)
    if not output_file.parent.exists():
        os.makedirs(output_file.parent)
    
    with output_file.open('w') as f:
        f.write('HIERARCHY\n')
        write_header(writer=f, node=header.root, level=0)
        
        f.write('MOTION\n')
        f.write(f'Frames: {len(channels)}\n')
        f.write(f'Frame Time: {1 / frame_rate}\n')

        for channel in channels:
            f.write(' '.join([f'{element}' for element in channel]) + '\n')
