![thumbnail](https://github.com/user-attachments/assets/b0aa865c-416c-4a3a-92be-56a1a77c8f4e)
# RenAIssance
The analysis of historical documents is a critical yet costly method in the Humanities. To reduce these costs, AI technology, specifically OCR (Optical Character Recognition), has started to be utilized. However, for many years, there was a lack of accurate OCR tools for Spanish documents from the Renaissance period, despite their academic importance. To address this issue, the HumanAI Foundation launched the **RenAIssance** project, where contributors implement accurate OCR models using various approaches.

# Dataset 
![letters](https://github.com/user-attachments/assets/c10584db-8f68-4897-a6c4-c70411ed9515)

The dataset used to train these models consists of images of printed documents from the target era, collected from diverse sources. A portion of the data has been manually labeled by RenAIssance mentors, who are experts in Spanish historical documents. The following printing irregularities in the data present challenges for creating high-accuracy OCR models:

- **Interchangeable Characters:** Characters such as 'u' and 'v', and 'f' and 's' were often used interchangeably.
- **Tildes and Diacritical Marks:** Used to save space or due to the reuse of type molds.
- **Old Spellings and Modern Interpretations:** Variations in character usage between historical and modern Spanish.
- **Line-End Hyphens:** Words split across lines were not always hyphenated.

Additionally, the deterioration and unique layouts of historical documents further complicate OCR tasks, making content extraction from images difficult.

# Method  
To address these challenges, contributors have introduced various state-of-the-art (SOTA) methods. These can be broadly classified into the following three approaches:

1. **CRNN Approach**  
2. **Vision Transformer Approaches**  
3. **Self-Supervised Learning Approach**  

All models, regardless of the approach used, achieve over 90% accuracy. For more detailed information on each approach, please refer to the contributors' repositories.


# 🤖 RenAIssance

RenAIssance is a HumanAI Foundation initiative designed to revolutionize how humans and AI collaborate. This project explores the boundaries of Artificial Intelligence for social good, aiming to build ethical, impactful, and open AI tools.

---

## 📌 About RenAIssance

RenAIssance is an open-source project focused on democratizing AI development by enabling ethical, explainable, and human-aligned AI systems. The goal is to build systems that enhance human capabilities while being transparent and socially responsible.

---

## ✨ Features

- 🧠 Human-centric AI systems
- 🔍 Research-based architecture and implementations
- 🌍 Open collaboration and community-driven innovation
- 🧪 Tools and datasets for ethical AI experiments

---

## 🧰 Tech Stack

- `Python`
- `PyTorch` / `TensorFlow` (depending on modules)
- `Node.js` / `React` (for frontend dashboards)
- `Docker` (for containerization)
- `GitHub Actions` (for CI/CD)

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure you have the following installed:
- `Git`
- `Node.js` and `npm`
- `Python 3.8+`
- `pip` or `conda`
- `Docker` (optional but recommended)

---

### 📥 Clone the Repository

```bash
git clone https://github.com/humanai-foundation/RenAIssance.git
cd RenAIssance
```

---

### 🛠️ Install Dependencies

```bash
# For Python backend
pip install -r requirements.txt

# For JavaScript frontend (if applicable)
cd frontend
npm install
```

---

### ▶️ Run the Project

```bash
# Run backend
python app.py  # or main.py depending on the entry point

# Run frontend
cd frontend
npm start
```

---

## 📂 Project Structure

```
RenAIssance/
│
├── backend/                # Python backend code
├── frontend/               # React or Node.js frontend
├── docs/                   # Documentation and research papers
├── datasets/               # AI/ML datasets used in the project
├── tests/                  # Unit and integration tests
└── README.md               # You are here!
```

---

## 🙋‍♂️ How to Contribute

We welcome all kinds of contributions!

### 🧩 Steps to Contribute:

1. **Fork the repo**
2. **Clone your fork**

   ```bash
   git clone https://github.com/your-username/RenAIssance.git
   cd RenAIssance
   ```

3. **Create a new branch**

   ```bash
   git checkout -b your-feature-name
   ```

4. **Make your changes**
5. **Commit and push**

   ```bash
   git add .
   git commit -m "Describe your changes"
   git push origin your-feature-name
   ```

6. **Open a Pull Request** with a meaningful title and description

---

## 🧠 Good First Issues

Want to contribute but not sure where to start? Look for issues labeled **`good first issue`** or **`help wanted`** in the [Issues section](https://github.com/humanai-foundation/RenAIssance/issues).

---

## 🤝 Community & Support

- Join our [Discord](https://discord.gg/humanai) for help and collaboration.
- Follow [HumanAI Foundation](https://humanai.org) for updates.
- Reach out via GitHub Discussions or Issues.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌟 Star this repo!

If you find this project helpful or inspiring, please give it a ⭐ on GitHub to show your support!