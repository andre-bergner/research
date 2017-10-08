
# just 2d image
a = np.array([[1,2,3],[4,5,6]])
np.stack([np.stack([a] + 2*[np.zeros([2,3])]), np.zeros([3,2,3])]).transpose(2,1,3,0).reshape(6,6)


# with batches, size = 2
a = np.array([ [[1,2,3],[4,5,6]] , [[9,8,7],[8,7,6]] ]) 
np.stack([np.stack([a] + 2*[np.zeros([2,2,3])]), np.zeros([3,2,2,3])]).transpose(2,3,1,4,0).reshape(2,6,6)


# with features, size = 1
a = np.array([ [[[1],[2],[3]], [[4],[5],[6]]] , [[[9],[8],[7]],[[8],[7],[6]]] ])
np.stack([np.stack([a] + 2*[np.zeros([2,2,3,1])]), np.zeros([3,2,2,3,1])]).transpose(2,3,1,4,5,0).reshape(2,6,6,1)