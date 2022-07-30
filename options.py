class Animals_Option(object):
    template_type = "SPHERE"
    bottleneck_size = 1024 
    overfit = True
    number_points = 15000
    nb_primitives = 1 # number_points/nb_primitives = nb_pts_in_primitive

    number_points_eval = 15000
    num_layers = 2
    remove_all_batchNorms = 0
    hidden_neurons = 512
    activation = 'relu'
    SVR = True
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    lrate = 0.001
    batch_size = 6
    print_every_n = 1
    validate_every_n = 20
    max_epochs =400

    def __init__(self):
        self.dim_template = self.dim_template_dict[self.template_type]


class Headpose_Option(object):
    template_type = "SPHERE"
    bottleneck_size = 1024 
    overfit = True
    number_points = 15000
    nb_primitives = 1 # number_points/nb_primitives = nb_pts_in_primitive

    number_points_eval = 15000
    num_layers = 2
    remove_all_batchNorms = 0
    hidden_neurons = 512
    activation = 'relu'
    SVR = True
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    lrate = 0.001
    batch_size = 16
    print_every_n = 1
    validate_every_n = 10
    max_epochs =500

    def __init__(self):
        self.dim_template = self.dim_template_dict[self.template_type]

class Headpose_normal_Option(object):
    template_type = "SPHERE"
    bottleneck_size = 1024 
    overfit = True
    number_points = 15000
    nb_primitives = 1 # number_points/nb_primitives = nb_pts_in_primitive

    number_points_eval = 15000
    num_layers = 2
    remove_all_batchNorms = 0
    hidden_neurons = 512
    activation = 'relu'
    SVR = True
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    lrate = 0.001
    batch_size = 6
    print_every_n = 1
    validate_every_n = 10
    max_epochs =400

    def __init__(self):
        self.dim_template = self.dim_template_dict[self.template_type]

class OptionSquare(object):
    template_type = "SQUARE"
    bottleneck_size = 1024 
    overfit = True
    number_points = 15000
    nb_primitives = 50 # number_points/nb_primitives = nb_pts_in_primitive

    number_points_eval = 15000
    num_layers = 2
    remove_all_batchNorms = 0
    hidden_neurons = 512
    activation = 'relu'
    SVR = True
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    lrate = 0.001
    batch_size = 8
    print_every_n = 1
    validate_every_n = 10
    max_epochs = 200

    def __init__(self):
        self.dim_template = self.dim_template_dict[self.template_type]
