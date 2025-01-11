from experiment.config import ConfigTrain

from schnetpack.datasets import QM9
import schnetpack.transform as transform

def construct_dataloaders(config: ConfigTrain):

    transforms = [
            transform.SubtractCenterOfMass(),
            transform.MatScipyNeighborList(config.radius),
            transform.CastTo32(),
        ]
    dataset = QM9(
            "./qm9.db",
            num_train=config.train_size,
            num_val=config.validation_size,
            batch_size=config.batch_size,
            transforms=transforms,
            remove_uncharacterized=True,
            num_workers=config.num_workers,
            split_file=config.split_file,
            #pin_memory=False,
            #load_properties=[config.prop],
    )
    dataset.prepare_data()
    dataset.setup()

    return dataset.train_dataloader(), dataset.test_dataloader(), dataset.val_dataloader()

def compute_mean_std(dataloader, key):

    n = 0
    total = 0

    for item in dataloader:
        item = item[key]
        n += len(item)
        total += item.sum()

    mean = total/n

    std = 0

    for item in dataloader:
        item = item[key]
        std += ((mean - item) ** 2).sum()

    std = (std/n)**0.5
    return mean, std


if __name__ == "__main__":
    from experiment.config import BuilderConfigExperiment
    from transformers import set_seed
    set_seed(0)

    config = (
        BuilderConfigExperiment()
        .set_vocab_dim(20)
        .set_feature_dim(256)
        .set_train_size(100_000)
        .set_validation_size(15_000)
        .set_batch_size(1024)
        .set_num_workers(6)
        .set_radius(3.)
        .set_path("./data/qm9.db")
        .set_split_file("./data/split2.npz")
        .build()
    )

    train, test, val = construct_dataloaders(config.train)
    print(compute_mean_std(train, "energy_U0"))
    import sys
    sys.exit(0)

    data_point = (next(iter(train)))
    print(data_point)

    print("I")
    print(f"i: {data_point['_idx_i']}")
    print(f"i: {data_point['_idx_i_local']}")
    print(data_point['_idx_i'] == data_point['_idx_i_local'])

    print("J")
    print(f"j: {data_point['_idx_j']}")
    print(f"j: {data_point['_idx_j_local']}")
    print(data_point['_idx_j'] ==data_point['_idx_j_local'])
