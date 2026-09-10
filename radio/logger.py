import csv
import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs", filename=None, fieldnames=None, float_precision=3):
        os.makedirs(log_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.csv"
        
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = fieldnames or ["Timestamp", "Epoch", "Train loss", "Valid loss", "Train ROC_AUC", "Valid ROC_AUC"]
        self.float_precision = float_precision
        
        # append (keep history) when resuming a run, else start fresh with a header
        existed = os.path.isfile(self.filepath) and os.path.getsize(self.filepath) > 0
        self.file = open(self.filepath, mode="a" if existed else "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not existed:
            self.writer.writeheader()
        self.file.flush()
    
    def _format_value(self, value):
        if isinstance(value, float):
            return round(value, self.float_precision)
        return value

    def log(self, metrics: dict):
        """
        metrics: dict like {"Epoch": 1, "Train loss": 0.23, "accuracy": 0.91}
        """
        metrics = metrics.copy()
        metrics["Timestamp"] = datetime.now().isoformat()
        
        # Format values
        formatted_metrics = {
            key: self._format_value(val)
            for key, val in metrics.items()
        }
        # Ensure all fields exist
        for field in self.fieldnames:
            if field not in formatted_metrics:
                formatted_metrics[field] = None

        self.writer.writerow(formatted_metrics)
        self.file.flush()

    def close(self):
        self.file.close()
        
