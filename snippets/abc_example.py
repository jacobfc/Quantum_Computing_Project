# This is an example for abc in python3
# For more detail see https://docs.python.org/3/library/abc.html 
import abc


# This specifies the 'interface'
class Plant(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def water(self):
        """ Put documentation here. """
        return


# An abstracted method taking a list of plants
def water_plants(plants):
    """ Water all plants in list plants. """
    for plant in plants:
        # if you explicitly want to check it's a plant indeed
        assert isinstance(plant, Plant), str(type(plant)) + "is not a Plant"
        # This would work even if Plant did not have the ABCMeta metaclass.
        # However, using abc allows the use of annotations and makes clear
        # Plant is an abstract class; having Plant as super class only
        # clearly specifies the interface, no functionality is inherited
        # (although, if wanted, functionality can be inherited with calls
        # to super()).

        plant.water()


class Tree(Plant):
    def __init__(self, tree_type):
        """ Create a new tree.

        Args:
            tree_type: String specifying typ of tree (e.g. "oak").
        """
        self.tree_type = tree_type

    def water(self):
        """ Water the tree """
        print("Watering a tree of type " + self.tree_type)


class Grass(object):
    pass
    def water(self):
        print("Watering grass")


# registering a subclass can also be done manually
Plant.register(Grass)


# trying to instantiate this leads to an error as water is annotated as
# @abstractmethod in Plant
class Sunflower(Plant):
    pass


if __name__ == '__main__':
    plants = [Grass(), Tree("oak"), Tree("birch")]

    water_plants(plants)

    # This will cause an error, see definition of Sunflower
    # sunflower = Sunflower()


