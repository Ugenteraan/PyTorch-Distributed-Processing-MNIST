# Distributed and Parallel Training using PyTorch

### The idea behind the concept

++**Memory Types**++
i) Pageable Memory
- A.K.A. Unpinned memory.
- When data is read and loaded into the RAM of a host machine, it is by default stored in the pageable memory space. This memory space can be paged out/paged in by other processes.
- A GPU cannot access the pageable memory space.
- Therefore, in order for a GPU to access the desired data, the data have to be transferred from the pageable memory space to page-locked memory space first.

ii) Page-Locked Memory
- A.K.A. Pinned memory.
- This space can be considered as a staging area before the data can be transferred from RAM to GPU.
- All the data in this space is locked and not allowed to be paged out by any other processes.
- The process of transferring data from pageable to page-locked space can be a computationally expensive process. Hence, normally the process is done in a multi-threaded fashion. Instead of waiting for the entire data to load and then begin the transfer from one space to another, there will be a separate thread that transfer the data to the pinned memory at the same time the data is being loaded into the unpinned memory.
- The GPU uses the PCIe connection on the motherboard to load the data into it. The better the PCI, the better the transfer rate and throughput.

++**Distributed and Parallel Training**++
PyTorch offers 2 types of ways to do the parallel training. The first module is called DataParallel while the second one is called DistributedDataParallel. The concept behind both of the methods are shown below.

++**DataParallel**++

- There will be a master GPU that distributes the mini-batches to other GPUs (it can be over a network to different host machines).
- The same model will be replicated across the GPUs.
- Run forward pass on every GPU.
- Gather all the outputs on the master GPU and calculate the loss.
- Distribute the loss back to all GPUs and calculate the gradient of every parameter.
- Average all the calculated gradients in the master GPU and update the parameter in the master GPU.
- Sync the model parameters across the GPUs. 

There are a few inefficiencies in this first approach.
- Redundancy when the data is broke down to mini-batches and scattered across the GPUs. Instead, this could have been done more efficiently by directly copying the data to those GPU.
- Models have to be synced at the beginning of every forward pass since the update happens at the master GPU.
- Since the forward pass is done in a multi-threaded fashion, the creation and destruction of the threads are overheads.
- The gradient averaging happens after the entire gradients in all the layers are calculated.

++**DistributedDataParallel**++
- The distributed minibatch sampler ensures that each process that runs in different GPU loads the data directly from the page-locked memory and that each process loads non-overlapping data.
- Every GPU will have identical model that runs the forward-pass on their respective mini-batch data.
- The loss and the gradient of the parameters calculation happens on the individual GPUs. Assuming that a neural network has _l_ layers (layer 1, layer 2, ..., layer _l_), the gradient calculation starts from layer _l_. Then the gradients from layer _l_ is used to calculate the gradients in layer _l_ -1 and so on. In order to be more efficient, after a process in any one of the GPU calculates the gradient in one of the layer, the gradients are immediately broadcasted to all the other GPUs in a multi-threaded fashion while the main thread continues to calculate the gradients in the following layer.
- The average of the gradients will be computed as soon as all the GPUs have broadcasted their gradients at a certain layer. 
- Since the models were identical and the gradients were kept in sync, the parameters are updated by the individual GPUs.

### Using PyTorch's DataDistributedParallel (DDP) module

**++ProcessGroup++**
- All the GPUs must communicate in some way to be in sync. Pytorch comes with a communication package that currently supports 3 backends at the time this note is written. **GLOO**, **MPI**, and **NCCL** are the 3 backends. The rule of thumb is that for CPU distributed processing, GLOO is to be used. For CPU distributed processing with InfiniBand, MPI is to be used and finally for GPU distributed processing, NCCL is to be used. During the inialization of this process group, it is vital that the number of total GPUs, the rank of each GPU and the master node initialized. 

++**Construction**++
- The DDP constructor broadcast the `state_dict()` of the model that runs in the master node's GPU 0 to all the other processes that runs in different GPUs. This is to make sure that all the processes start with the same exact replica of the model.
- Each DDP process also creates a local `Reducer` that is responsible for gradient synchronization during the back propagation.

**++Forward Propagation++**
- During the forward propagation, the DDP also analyzes the output if `find_unused_parameter` is set to True. This mode allows certain parameters to not be updated.

**++Back Propagation++**
- The loss on the local models are used to calculate the parameter gradients by their own respective GPU. There are hooks registered during the construction time that gets triggered when gradients are calculated at a certain layer. The DDP then uses this trigger to spin up a different thread to synchronize the gradients across all the GPUs.
- Until the gradients from all the processes are ready, the `allreduce` operation will wait. Once the gradients are ready, the gradients will be averaged and written directly to the `param.gradd field in all parameters.

**++Optimization++**
The Optimizer sees optimizing the model as a local process since all the models are in sync and there is no need for further communication among the models.

The DDP module, when used on multiple nodes that are in the same network, it expects that the training scripts are explicitly launched on every node manually. Since the module has the information on how many nodes and GPUs are there, the training would not start until all the GPUs are present.

**Source**
1) https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/
2) https://pytorch.org/docs/master/notes/ddp.html
3) https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html#torch.nn.DataParallel
4) https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html


