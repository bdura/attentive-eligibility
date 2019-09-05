from .minigrid import *


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class EmptyEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5)


class CustomEmptyEnv(MiniGridEnv):

    def __init__(
            self,
            height=5,
            width=3,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            noise=None,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.noise = noise

        super().__init__(
            height=height,
            width=width,
            max_steps=4 * height * width,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Re-seed to have a consistent environment
        np.random.seed(0)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height, noise=self.noise)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class CustomEmptyEnv3x6(CustomEmptyEnv):
    def __init__(self):
        super().__init__(height=3, width=6)


class CustomEmptyEnv3x8(CustomEmptyEnv):
    def __init__(self):
        super().__init__(height=3, width=8)


class CustomEmptyEnv4x8(CustomEmptyEnv):
    def __init__(self):
        super().__init__(height=4, width=8)


class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)


class EmptyEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6)


class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)


class EmptyEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16)
