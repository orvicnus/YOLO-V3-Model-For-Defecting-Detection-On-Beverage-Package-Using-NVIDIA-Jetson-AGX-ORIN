def get_loaders():
  train_dataset = YOLODataset(
      img_dir=IMG_TRAIN_DIR,
      label_dir=LABEL_TRAIN_DIR,
      anchors=ANCHORS,
      S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
      transform=train_transforms,
  )
  test_dataset = YOLODataset(
      img_dir=IMG_TEST_DIR,
      label_dir=LABEL_TEST_DIR,
      anchors=ANCHORS,
      S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
      transform=test_transforms,
  )
  train_loader = DataLoader(
      dataset=train_dataset,
      batch_size=BATCH_SIZE,
      num_workers=NUM_WORKERS,
      pin_memory=PIN_MEMORY,
      shuffle=True,
      drop_last=False,
  )
  test_loader = DataLoader(
      dataset=test_dataset,
      batch_size=BATCH_SIZE,
      num_workers=NUM_WORKERS,
      pin_memory=PIN_MEMORY,
      shuffle=False,
      drop_last=False,
  )

  train_eval_loader = DataLoader(
      dataset=train_dataset,  # Menggunakan train_dataset yang sama
      batch_size=BATCH_SIZE,
      num_workers=NUM_WORKERS,
      pin_memory=PIN_MEMORY,
      shuffle=False,
      drop_last=False,
  )

  return train_loader, test_loader, train_eval_loader