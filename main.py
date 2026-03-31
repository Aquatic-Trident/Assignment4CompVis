from kaggleLoad import *
from visualise import *

dataset, dataloader = Download()

visualize_batch(dataloader)
print(torch.cuda.is_available())