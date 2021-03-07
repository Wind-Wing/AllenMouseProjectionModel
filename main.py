from model.cortex_model import CortexModel
from model.global_model import GlobalModel

if __name__ == "__main__":
    global_model = GlobalModel()
    cortex_model = CortexModel(global_model)
    cortex_model.calc_regional_projection_matrix()
    # cortex_model.calc_region_projection_volume()
