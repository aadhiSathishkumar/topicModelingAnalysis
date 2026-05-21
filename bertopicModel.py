from bertopic import BERTopic
import re
import spacy
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Stopwords
stop_words = set(stopwords.words("english"))

# Common useless phrases in EE and CpE course catalogs
generic_phrases = [
    "this", "course",
    "students", "will",
    "introduction", "to",
    "topics", "include",
    "covers",
    "introduce",
    "designed",
    "provides", "an", "overview"
]

#Function: Preprocess text by removing generic phrases, stopwords, and lemmatizing
#Input: Raw course description text
#Output: Cleaned and preprocessed text ready for topic modeling
def preprocess_text(text):
# Lowercase (BERTopic is case-sensitive)
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Lemmatize using spaCy
    doc = nlp(text)

    words = []
    for token in doc:
        lemma = token.lemma_
        # Keep words that are not stop words or generic words and are longer than 2 characters
        if lemma not in stop_words and lemma not in generic_phrases and len(lemma) > 2:
            words.append(lemma)

    return " ".join(words)

courses = {
    "ELEC ENG 1010": "Discussion of current topics",
    "ELEC ENG 2100": "Circuit elements, signals, Kirchhoff's laws, network theorems, mesh and nodal analysis, transient and complete response of RL, RC, and RLC circuits",
    "ELEC ENG 2101": "Safety, basic measurements and meters, oscilloscopes, resistor networks, measurement of capacitors and inductors, RLC circuit response",
    "ELEC ENG 2120": "Analysis of steady state AC circuits, phasor notation, polyphase circuits, complex frequency and frequency response, magnetically coupled circuits",
    "ELEC ENG 2200": "Materials and device structures for applications in analog and digital electronics. Topics include characteristics and basic circuits for diodes, field-effect transistors, bipolar junction transistors, and operational amplifiers",
    "ELEC ENG 2201": "Laboratory tools and measurement techniques for basic electronic circuits using diodes, field-effect transistors, bipolar junction transistors, and operational amplifiers",
    "ELEC ENG 2410": "Analysis methods for continuous-time signals and systems in the time and frequency domains including signal models, Fourier transforms, and Laplace transforms. Examples of control and communication systems are included",
    "ELEC ENG 2411": "Laboratory and software tools for the analysis of linear systems. Topics include Matlab programming and spectral analysis",
    "ELEC ENG 2800": "Taught primarily as an alternating current circuits course with direct current circuits as a special case. Current, voltage, and power relations; complex algebra; voltage and power relations in polyphase circuits with an emphasis on transformers and induction machines",
    "ELEC ENG 3000": "Problems or readings on specific subjects or projects in the department",
    "ELEC ENG 3002": "On-the-job experience gained through cooperative education with industry with credit arranged through departmental cooperative advisor",
    "ELEC ENG 3100": "Diode and transistor circuits, small signal analysis, amplifier design, differential and operational amplifiers, flipflop circuits and waveshaping",
    "ELEC ENG 3101": "Experiments in design with diodes, transistors, differential and operational amplifiers, and logic components",
    "ELEC ENG 3120": "Continuation of electronics I with additional diode and transistor circuits, advanced logic components, and flipflop registers",
    "ELEC ENG 3121": "Experiments in advanced circuit design with diodes, power transistors, integrated circuits, and logic registers",
    "ELEC ENG 3250": "Semiconductor materials and devices for electronic and photonic applications including crystal physics and optoelectronic devices",
    "ELEC ENG 3320": "Laplace transforms, formulation of the control problem, system equations and models, time and frequency domain analysis, stability, and design of linear control systems",
    "ELEC ENG 3321": "Software tools for control systems analysis",
    "ELEC ENG 3340": "Introduction to programmable automation in manufacturing including programmable logic controller hardware and programming languages",
    "ELEC ENG 3410": "Analysis methods for discrete-time signals and systems in the time and frequency-domains including signal models and Fourier techniques",
    "ELEC ENG 3411": "Software tools for signal and system representation for time and frequency-domain analysis",
    "ELEC ENG 3430": "Signals and systems for digital communications including source formatting and digital pass-band modulation and demodulation",
    "ELEC ENG 3431": "Laboratory and software tools for analysis of communications and linear and nonlinear signals",
    "ELEC ENG 3440": "Continuation of digital communications with further study of signals and spectra with applications",
    "ELEC ENG 3500": "Magnetics and magnetically coupled circuits, electromechanical energy conversion, rotating magnetic fields and machines",
    "ELEC ENG 3501": "Experiments with power measurement, transformers, magnetically coupled circuits, and rotating magnetic fields",
    "ELEC ENG 3540": "Power system components and transmission lines with balanced power system theory, design and fault analysis",
    "ELEC ENG 3541": "Computer-aided analysis of voltage regulation, power flow, and economic analysis",
    "ELEC ENG 3600": "Static electric and magnetic fields using vector analysis and time-varying electromagnetic fields using Maxwell's equations",
    "ELEC ENG 4000": "Problems or readings on specific subjects or projects in the department",
    "ELEC ENG 4001": "This course is designed to give the department an opportunity to test a new course. Variable title",
    "ELEC ENG 4096": "A complete design cycle project with design, documentation, analysis, implementation and testing",
    "ELEC ENG 4097": "Continuation of senior project work",
    "ELEC ENG 4099": "Undergraduate research in electrical engineering",
    "ELEC ENG 4380": "Project work related to topics in automation engineering",
    "COMP ENG 1200": "Introduction to electronics and digital circuit design including combinational logic and sequential circuits using tools, logic gates and field programmable gate arrays",
    "COMP ENG 2210": "Core components of digital systems including binary numbers, truth tables, Boolean algebra, combinational and sequential logic",
    "COMP ENG 2211": "Introduction to digital design techniques, logic gates, flipflops, timing analysis, and programming of programmable logic devices",
    "COMP ENG 3000": "Problems or readings on specific subjects or projects in the department",
    "COMP ENG 3002": "On-the-job experience gained through cooperative education with industry",
    "COMP ENG 3110": "Basic concepts of computer organization and design including performance, computer arithmetic, memory hierarchy, and I/O controllers",
    "COMP ENG 3150": "Microcontroller-based digital system design covering machine organization, interface design, and programming for real-time embedded systems",
    "COMP ENG 3151": "Advanced digital design techniques with microcontroller-based designs and hardware–software codesign",
    "COMP ENG 4000": "Problems or readings on specific subjects or projects in the department",
    "COMP ENG 4001": "Variable title special topics course designed to give the department an opportunity to test a new course",
    "COMP ENG 4096": "A complete senior design cycle including design, documentation, analysis, implementation, and testing of a product",
    "COMP ENG 4097": "Continuation of the computer engineering senior project sequence",
    "COMP ENG 4099": "Undergraduate research in computer engineering",
    "COMP ENG 5110": "Advanced principles of computer architecture including pipeline, memory systems, and interconnection networks",
    "COMP ENG 5120": "Organization and design of modern digital computers focusing on processors, memory systems, and I/O units",
    "COMP ENG 5130": "Design of digital systems based on advanced microprocessors with hardware and software laboratory projects",
    "COMP ENG 5151": "Design of 32-bit microcontroller-based systems including instruction set architecture and peripherals",
    "COMP ENG 5160": "Development of hardware and software for real-time embedded systems including communication schemes and sensors",
    "COMP ENG 5170": "Introduction to real-time operating systems, scheduling algorithms, and R-T system design",
    "COMP ENG 5210": "Introduction to VLSI design including transistor-level design and timing analysis",
    "COMP ENG 5220": "Digital system modeling for simulation, synthesis, and rapid prototyping",
    "COMP ENG 5230": "Introduction to quantum computing principles and architectures",
    "COMP ENG 5310": "Introduction to computational intelligence including neural networks and evolutionary computing",
    "COMP ENG 5320": "Mathematical and computational foundations of game theory and its applications",
    "COMP ENG 5410": "Introduction to computer communication networks including architecture, protocols, and performance",
    "COMP ENG 5420": "Basic issues in cryptography and network security including encryption and authentication",
    "COMP ENG 5430": "Introduction to wireless networks covering transmission fundamentals and wireless protocols",
    "COMP ENG 5450": "Fundamentals of digital image processing including transforms and enhancement",
    "COMP ENG 5460": "Image filtering, pattern recognition, and boundary detection for machine vision",
    "COMP ENG 5510": "Design and analysis of fault-tolerant digital systems",
    "COMP ENG 5610": "Real-time digital signal processing with programmable DSP chips",
    "COMP ENG 5620": "Signal integrity analysis in high-speed digital and mixed-signal design",
    "COMP ENG 5803": "Mathematical logic with applications in computing and engineering",
    "COMP ENG 5820": "Integration of mechanical, electrical, and control systems for mechatronics",
    "COMP ENG 5880": "Introduction to robotics including kinematics, trajectories, and decision making",
}

course_names = list(courses.keys())
course_descriptions = list(courses.values())

# Apply preprocessing
cleaned_descriptions = [preprocess_text(desc) for desc in course_descriptions]

# Initialize BERTopic model with a minimum topic size of 2
topic_model = BERTopic(min_topic_size=2)

# Fit the model and transform the course descriptions into topics
topics, probs = topic_model.fit_transform(cleaned_descriptions)

# Reduce outliers by reassigning them to the most similar existing topic
topics = topic_model.reduce_outliers(cleaned_descriptions, topics)

# Update topics with the outliers reassigned
topic_model.update_topics(cleaned_descriptions, topics)

mapping = topic_model.visualize_topics()
mapping.show()

topic_keywords = topic_model.visualize_barchart()
topic_keywords.show()

topic_info = topic_model.get_topic_info()
print(topic_info)

for course, topic in zip(course_names, topics):
    print(f"{course}: Topic {topic}")

print("done")