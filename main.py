from kaggleLoad import *
from visualise import *
from ImgResize import *


dataset = Download()
train, test = Split(dataset, TEST_SPLIT)
train, val = Split(train, TRAIN_SPLIT)

train_loader = Loader(train)
val_loader = Loader(val)
test_loader = Loader(test)

visualize_batch(train_loader)


print(torch.cuda.is_available())
print(torch.version.cuda)

