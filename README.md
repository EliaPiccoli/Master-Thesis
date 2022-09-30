# Master-Thesis
## Repository for the code of my Master's Thesis @ University of Pisa

### Abstract
Reinforcement Learning in recent years has reached astonishing results exploiting huge and complex deep architectures. However, this has come at the cost of unsustainable computational efforts. A common characteristic of all state of art approaches, common in the majority of Machine Learning algorithms, is that the agent’s network learns to solve the task “from scratch”, that is from a randomized initialization, without reusing previously learned skills or doing it only to a very limited extent. In order to challenge the problem of transfer and reuse, we propose a new approach called Skilled Deep Q-Learning, which leverages pre-trained unsupervised skills as agents’ prior knowledge. In the first part of the work, we discuss the implementation of this approach comparing its performance using the Atari suite and investigate how the agent uses these skills. In the second part, we focus on Continual Reinforcement Learning scenarios, trying to extend the proposed approach in a setting where the Reinforcement Learning agent learns more than one game simultaneously. Finally, we present various research paths that can be explored to further develop, understand and improve the proposed approach.

---
### Code Structure
The repository reports the code for all the models using _PyTorch_. In particular, each folder - based on its name - contains the implementation of a particular architecture:
- [frame_synthesis](https://arxiv.org/pdf/1702.02463.pdf): Liu, Ziwei, et al. "Video frame synthesis using deep voxel flow." _Proceedings of the IEEE international conference on computer vision_. 2017.
- [keypoints_transporter](https://arxiv.org/abs/1906.11883): Kulkarni, Tejas D., et al. "Unsupervised learning of object keypoints for perception and control." _Advances in neural information processing systems_ 32 (2019).
- [progressive_nn](https://arxiv.org/abs/1606.04671): Rusu, Andrei A., et al. "Progressive neural networks." _arXiv preprint arXiv:1606.04671_ (2016).
- [state_representation](https://arxiv.org/abs/1906.08226): Anand, Ankesh, et al. "Unsupervised state representation learning in atari." _Advances in neural information processing systems_ 32 (2019).
- [video_object_segmentation](https://arxiv.org/abs/1805.07780): Goel, Vikash, Jameson Weng, and Pascal Poupart. "Unsupervised video object segmentation for deep reinforcement learning." _Advances in neural information processing systems_ 31 (2018).
