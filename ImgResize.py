from settings import *
import torchvision
import torchvision.transforms as T

def ComposeTransform():
    transforms = []
    box_transforms = []
    for transform_ in transform:
        transform_method, params = Unpack(transform_)
        
        match transform_method:
            case TransformMethod.resize:
                transforms.append(T.Resize(*params))
                box_transforms.append((ResizeBox, params))
            case TransformMethod.tensor:
                transforms.append(T.ToTensor(*params))
                box_transforms.append((Nothing, []))
        
    return T.Compose(transforms), box_transforms

def Nothing(img, bbox):
    return bbox

def ResizeBox(img, bbox, target_size):
    width_r = target_size[0] / img.width
    height_r = target_size[1] / img.height

    bbox[0,0] *= width_r
    bbox[0,1] *= height_r
    bbox[0,2] *= width_r
    bbox[0,3] *= height_r
    return bbox

TransformMethod = Enum('TransformMethod', [('resize', 1), ('tensor', 2)])

transform = [
    (TransformMethod.resize, [TARGET_SIZE]),
    TransformMethod.tensor
]

def Unpack(info):
    if isinstance(info, tuple):
        # unpacks '(enum, p)' where p can be anything, including a list
        if len(info) == 2:
            enum, params = info
        # turns '(enum, p1, p2, ... pn)' into the expected format
        else:
            enum = info[0]
            params = []
            for i in range(len(info) - 1):
                params.append(info[i+1])
    # turns 'enum' into '(enum, [])'
    else:
        enum = info
        params = []

    # turns '(enum, p)' into '(enum, [p1])'
    if not isinstance(params, list):
        params = [params]

    return (enum, params)
