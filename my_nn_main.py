import mnist_loader as mnldr
import my_network as mn
import my_network2 as mn2

training_data, validation_data, test_data = mnldr.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

# my_nn = mn2.Network([784, 30, 10], cost=mn2.QuadraticCost)
# my_nn.SGD(training_data, 10, 10, 1, lmbda=0.0001, 
		# evaluation_data=None,
		# monitor_evaluation_cost=False,
		# monitor_evaluation_accuracy=False)

# my_nn.save('my_nn_save1.json')

my_nn = mn2.load('my_nn_save1.json')
c = my_nn.total_cost(test_data, 0.0001, convert=True)
a = my_nn.accuracy(test_data)
print(c)
print(a)
