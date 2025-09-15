import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class BirdSpeciesGUI:
    def __init__(self, root, predictor, model_accuracy=None, training_time=None, best_accuracy=None):
        self.root = root
        self.predictor = predictor
        self.model_accuracy = model_accuracy
        self.training_time = training_time
        self.best_accuracy = best_accuracy
        self.root.title("Bird Species Classification")
        self.root.geometry("900x750")  # Increased window size for additional information
       
        self.create_widgets()
   
    def create_widgets(self):
        """Create GUI elements"""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
       
        # Application title
        title_label = tk.Label(main_frame, text="Bird Species Classification using EfficientNet",
                              font=("Arial", 18, "bold"), fg="darkblue")
        title_label.pack(pady=10)
       
        # Model information frame
        info_frame = tk.LabelFrame(main_frame, text="Model Information", font=("Arial", 12, "bold"),
                                  relief=tk.GROOVE, borderwidth=2)
        info_frame.pack(fill=tk.X, pady=10)
       
        # Display model accuracy and training time if available
        info_text = ""
        if self.model_accuracy is not None:
            info_text += f"Model Accuracy: {self.model_accuracy*100:.2f}%"
        if self.training_time is not None:
            info_text += f"   |   Training Time: {self.training_time:.2f} seconds"
        if self.best_accuracy is not None:
            info_text += f"\nBest Validation Accuracy: {self.best_accuracy*100:.2f}%"
       
        if info_text:
            model_info_label = tk.Label(info_frame, text=info_text, font=("Arial", 10),
                                       fg="green" if self.model_accuracy and self.model_accuracy > 0.8 else "orange")
            model_info_label.pack(pady=5)
        else:
            no_model_label = tk.Label(info_frame, text="Model not trained yet", font=("Arial", 10), fg="red")
            no_model_label.pack(pady=5)
       
        # Button to load image
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
       
        load_btn = tk.Button(button_frame, text="Load Image and Classify",
                            command=self.load_and_predict,
                            width=20, height=2, bg="lightblue", font=("Arial", 12, "bold"))
        load_btn.pack(pady=10)
       
        # Frame for image and results
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
       
        # Image frame on the left
        left_frame = tk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
       
        image_title = tk.Label(left_frame, text="Selected Image", font=("Arial", 12, "bold"))
        image_title.pack(pady=5)
       
        self.image_frame = tk.Frame(left_frame, width=350, height=350,
                                   relief=tk.SUNKEN, borderwidth=2, bg="white")
        self.image_frame.pack(pady=5)
        self.image_frame.pack_propagate(False)
       
        self.image_label = tk.Label(self.image_frame, text="Image will be displayed here",
                                   font=("Arial", 10), fg="gray")
        self.image_label.pack(expand=True)
       
        # Results frame on the right
        right_frame = tk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
       
        result_title = tk.Label(right_frame, text="Classification Results",
                               font=("Arial", 12, "bold"))
        result_title.pack(pady=5)
       
        # Text frame with scrollbar
        text_frame = tk.Frame(right_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
       
        self.result_text = tk.Text(text_frame, height=15, width=50,
                                  font=("Arial", 11), wrap=tk.WORD)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
       
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=
        self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
       
        # Set initial text
        initial_text = "Welcome! This application can classify 200 different bird species.\n\n"
        initial_text += "1. Click the 'Load Image and Classify' button\n"
        initial_text += "2. Select a bird image from your device\n"
        initial_text += "3. View the classification results here\n\n"
        initial_text += "The bird species and confidence level will appear here."
       
        self.result_text.insert(tk.END, initial_text)
        self.result_text.config(state=tk.DISABLED)  # Make text read-only
       
        # Footer frame
        footer_frame = tk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=10)
       
        footer_label = tk.Label(footer_frame, text="Developed using EfficientNet and TensorFlow",
                               font=("Arial", 9), fg="gray")
        footer_label.pack()
   
    def load_and_predict(self):
        """Load image and classify"""
        try:
            # Open file dialog to select image
            file_path = filedialog.askopenfilename(
                title="Select Bird Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
           
            if file_path:
                # Display image
                img = Image.open(file_path)
                img.thumbnail((340, 340))  # Appropriate size for the frame
                photo = ImageTk.PhotoImage(img)
               
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
               
                # Enable results field for writing
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
               
                # Add loading message
                self.result_text.insert(tk.END, "Processing image and classifying bird...")
                self.root.update()  # Update interface immediately
               
                # Classify image
                results = self.predictor.predict_image(file_path)
               
                # Display results
                result_str = "="*50 + "\n"
                result_str += "Bird Classification Results\n"
                result_str += "="*50 + "\n\n"
               
                result_str += f"🎯 Predicted Bird: {results['predicted_class']}\n\n"
                result_str += f"📊 Confidence Level: {results['confidence']*100:.2f}%\n\n"
               
                result_str += "-"*30 + "\n"
                result_str += "Top 5 Predictions:\n"
                result_str += "-"*30 + "\n"
               
                for i, (class_name, confidence) in enumerate(results['top_5']):
                    result_str += f"{i+1}. {class_name}: {confidence*100:.2f}%\n"
               
                # Add some formatting based on confidence level
                if results['confidence'] > 0.7:
                    result_str += "\n✅ High Confidence Classification"
                elif results['confidence'] > 0.4:
                    result_str += "\n⚠️  Medium Confidence Classification"
                else:
                    result_str += "\n❌ Low Confidence Classification"
               
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, result_str)
               
                # Color results based on confidence level
                self.highlight_results(results['confidence'])
               
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the image: {str(e)}")
        finally:
            self.result_text.config(state=tk.DISABLED)
   
    def highlight_results(self, confidence):
        """Color results based on confidence level"""
        # Determine background color based on confidence level
        if confidence > 0.7:
            bg_color = "#e6ffe6"  # Light green
        elif confidence > 0.4:
            bg_color = "#fff9e6"  # Light orange
        else:
            bg_color = "#ffe6e6"  # Light red
       
        # Apply background color
        self.result_text.config(bg=bg_color)
       
        # Text formatting
        self.result_text.tag_configure("bold", font=("Arial", 12, "bold"))
        self.result_text.tag_configure("big", font=("Arial", 14, "bold"))
        self.result_text.tag_configure("green", foreground="darkgreen")
        self.result_text.tag_configure("red", foreground="darkred")
       
        # Apply formatting to text parts
        self.result_text.tag_add("bold", "4.0", "4.end")
        self.result_text.tag_add("big", "4.0", "4.20")
       
        if confidence > 0.7:
            self.result_text.tag_add("green", "4.0", "4.end")
        elif confidence < 0.4:
            self.result_text.tag_add("red", "4.0", "4.end")