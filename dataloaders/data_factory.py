import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_activity import ActivityNetMeDataLoader
from dataloaders.dataloader_charades import CharadesMeDataloader
from dataloaders.mydataloader_charades import MyCharadesMeDataloader
from dataloaders.shuffle_dataloader_charades import ShuffleCharadesMeDataloader
from dataloaders.shuffle_dataloader_activitynet import ShuffleActivityNetMeDataLoader
from dataloaders.mydataloader_activity import MyActivityNetMeDataLoader

def dataloader_factory(args, tokenizer, logger):
    assert args.datatype in DATALOADER_DICT
    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None
    
    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    
    train_dataloader, train_length, train_sampler = None, 0, None
    if args.do_train:
        # load the data(create corresponding dataloader)
        if args.shuffle_exp:
            print("shuffle experiment")
            train_dataloader, train_length, train_sampler = DATALOADER_DICT2[args.datatype]["train"](args, tokenizer)
        else:
            print("no shuffle experiment")
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
    
    return train_dataloader, test_dataloader, train_length, test_length, train_sampler


def mydataloader_activity_train(args, tokenizer):
    activity_dataset = MyActivityNetMeDataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        K = args.K,
        fps = args.K
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activity_dataset), train_sampler

def shuffle_dataloader_activity_train(args, tokenizer, subset="train"):
    print("shuffle dataloader")
    activity_dataset = ShuffleActivityNetMeDataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        shuffle_events=args.shuffle_events
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(activity_dataset), train_sampler

def dataloader_activity_test(args, tokenizer, subset="test"):
    activity_testset = ActivityNetMeDataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        activity_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(activity_testset)


def dataloader_charades_train(args, tokenizer):
    dataset = CharadesMeDataloader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )
    return dataloader, len(dataset), sampler

def dataloader_charades_test(args, tokenizer, subset="test"):
    dataset = CharadesMeDataloader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(dataset)

# modified
def mydataloader_charades_train(args, tokenizer):
    print("normal dataloader")
    dataset = MyCharadesMeDataloader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        K = args.K,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )
    return dataloader, len(dataset), sampler

def shuffle_dataloader_charades_train(args, tokenizer):
    print("shuffle dataloader")
    dataset = ShuffleCharadesMeDataloader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        shuffle_events=args.shuffle_events
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )
    return dataloader, len(dataset), sampler

# modified charades part
DATALOADER_DICT = {"activity": {"train": mydataloader_activity_train, "val": dataloader_activity_test, "test": None},
                   "charades": {"train": mydataloader_charades_train, "val": dataloader_charades_test,
                                "test": dataloader_charades_test}}
DATALOADER_DICT2 = {"activity": {"train": shuffle_dataloader_activity_train, "val": dataloader_activity_test, "test": None},
                   "charades": {"train": shuffle_dataloader_charades_train, "val": dataloader_charades_test,
                                "test": dataloader_charades_test}}

