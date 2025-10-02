train_loader, test_loader, train_eval_loader = get_loaders()

# import config
import torch
import torch.optim as optim

# from model import YOLOv3
from tqdm import tqdm
# from utils import (
#     mean_average_precision,
#     cells_to_bboxes,
#     get_evaluation_bboxes,
#     save_checkpoint,
#     load_checkpoint,
#     check_class_accuracy,
#     get_loaders,
#     plot_couple_examples
# )
# from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

if os.path.exists(CHECKPOINT_FILE):
    LOAD_MODEL = False

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y0, y1, y2 = (
            y[0].to(DEVICE),
            y[1].to(DEVICE),
            y[2].to(DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=f"{mean_loss:.6f}")

def evaluate_fn(test_loader, model, loss_fn, scaled_anchors):
    model.eval()
    losses = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y0, y1, y2 = (
                y[0].to(DEVICE),
                y[1].to(DEVICE),
                y[2].to(DEVICE),
            )

            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
            losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    model.train()
    return mean_loss

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=6
):
    average_precisions = []
    all_precisions = []
    all_recalls = []
    tp_fp_fn_stats = []

    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Mendapatkan deteksi dan ground truths untuk kelas saat ini
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # Inisialisasi tensor untuk melacak deteksi setiap ground truth
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Urutkan deteksi berdasarkan confidence
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        # Evaluasi setiap deteksi
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            best_gt_idx = -1  # ubah inisialisasi ke -1 untuk menghindari error

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # Menghitung FN
        FN = total_true_bboxes - TP.sum().item()
        # TN diasumsikan sebagai semua deteksi lainnya untuk kelas selain kelas saat ini
        TN = len(pred_boxes) - (TP.sum().item() + FP.sum().item() + FN)

        precision_per_class = TP.sum().item() / (TP.sum().item() + FP.sum().item() + epsilon)
        recall_per_class = TP.sum().item() / (TP.sum().item() + FN + epsilon)

        # Simpan statistik
        tp_fp_fn_stats.append({
            "class": c,
            "TP": TP.sum().item(),
            "FP": FP.sum().item(),
            "FN": FN,
            "TN": TN,
            "Precision": precision_per_class,
            "Recall": recall_per_class
        })

        all_precisions.append(precision_per_class)
        all_recalls.append(recall_per_class)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    mean_average_precision = sum(average_precisions) / len(average_precisions)
    mean_precision = sum(all_precisions) / len(all_precisions)
    mean_recall = sum(all_recalls) / len(all_recalls)

    return mean_average_precision, mean_precision, mean_recall, tp_fp_fn_stats

def main():
    seed_everything(seed=42)

    model = YOLOv3(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_FILE, model, optimizer, LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        test_loss = evaluate_fn(test_loader, model, loss_fn, scaled_anchors)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Test Loss: {test_loss}")

        if SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f'/content/drive/MyDrive/Skripsi/checkpoint_model.pth2.tar')

        if epoch > 0 and epoch % 1 == 0:
            print(f"On Test Loader (Epoch {epoch + 1}/{NUM_EPOCHS}):")
            check_class_accuracy(model, test_loader, threshold=CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=NMS_IOU_THRESH,
                anchors=ANCHORS,
                threshold=CONF_THRESHOLD,
            )
            mapval, mean_precision, mean_recall, tp_fp_fn_stats = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            print(f"Average Precision: {mean_precision}")
            print(f"Average Recall: {mean_recall}")

            for stats in tp_fp_fn_stats:
                print(f"Class {stats['class']} - TP: {stats['TP']}, FP: {stats['FP']}, TN: {stats['TN']}, FN: {stats['FN']}")
                print(f"Class {stats['class']} - Precision: {stats['Precision']}, Recall: {stats['Recall']}")

if __name__ == "__main__":
    main()
