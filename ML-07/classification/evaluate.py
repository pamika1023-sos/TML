import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def plot_history(history, title="Training History"):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    
    # บันทึกไฟล์กราฟ (ลบอักขระพิเศษเพื่อไม่ให้เกิด error ตอนตั้งชื่อไฟล์)
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{safe_title}_plot.png"
    plt.savefig(filename)
    plt.close() # ปิด plot เพื่อประหยัดเมมโมรี่และเคลียร์หน้ากระดาษ