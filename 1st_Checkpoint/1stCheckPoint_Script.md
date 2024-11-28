## Carletto

## Intro (10 seconds)

Good morning/afternoon everyone. Today, my team and I will walk you through our project on oil spill detection in seawater using Synthetic Aperture Radar, or SAR. We aim to tackle a critical environmental issue with advanced technology. Let’s dive into the specifics.

_Good morning/afternoon everyone! My team and I are excited to present our project on oil spill detection in the ocean using SAR imagery and neural networks. During the next 10 minutes we’re going to walk you through how we’re tackling this important environmental issue with cutting-edge technology._

###  Slide 1: The Problem (30 seconds)

Oil spills are a major environmental disaster with severe impacts on marine life, local economies, and the overall ecosystem. Traditional detection methods, like visual inspections and sampling, are time-consuming, costly, and often limited by conditions such as weather and visibility. This is where advanced techniques come into play, specifically semantic segmentation. Semantic segmentation allows us to classify each pixel in an image, enabling precise identification of oil spill regions within SAR images. This level of precision is critical in differentiating oil spills from other sea features, offering a powerful tool in challenging open-sea environments.

_Oil spills are a major threat to marine ecosystems. They can have devastating impacts on sea life, coastal communities, and the environment as a whole. Traditional methods for detecting spills, like manual inspections, can be slow and are often affected by weather or light conditions._

_With our project, we’re focusing on continuous monitoring and precise detection, which are key to managing these spills effectively. We use semantic segmentation with Synthetic Aperture Radar, or SAR, images—provided by the European Space Agency—to identify oil spills accurately. This segmentation process lets us analyze each pixel in the image, making it possible to distinguish oil spills from other features in the sea._

### Slide 2: Value Proposition (30 seconds)

Our project leverages Synthetic Aperture Radar, or SAR, for oil spill detection. SAR technology allows us to capture images regardless of weather or light conditions, offering a consistent and reliable way to monitor large marine areas. With this technology, we aim to provide faster and more accurate detection of oil spills, enabling quicker response times, reducing environmental impact, and potentially lowering cleanup costs. By combining technology with the Sustainable Development Goals, we’re aiming for a meaningful solution to a critical problem.

_Our project offers several key benefits that align with the UN Sustainable Development Goals. First, environmental impact: by detecting spills faster, we can reduce the harm they cause. Next is response efficiency: early detection means we can mobilize resources quickly, limiting the spread of oil._

_And lastly, scalability: this technology can be deployed across various marine environments globally, wherever SAR data is available. So, by using SAR imaging combined with neural networks, we’re building a system that’s not just accurate and timely, but also adaptable for widespread use._

### Slide 3: Objectives (45 seconds)

Our project has four main objectives, each designed to advance oil spill detection using SAR imagery. First, we aim to analyze the SAR dataset to understand its structure and identify potential challenges in detection. Second, we’ll implement and test a variety of neural network models tailored for segmentation tasks, with the goal of identifying oil spills accurately.

Next, we evaluate each model’s performance and finally, the selected model will be deployed in a real-time monitoring system.

These objectives bring clear benefits: increased accuracy due to advanced machine learning techniques, greater efficiency by reducing detection and response times, significant contributions to environmental monitoring research, and the ability for consistent, reliable monitoring. Together, these benefits make our project a strong solution for sustainable ocean protection and rapid oil spill response.

_We have four main objectives. First, we start with analyzing the SAR dataset to understand its structure and find any challenges. Next, we’ll test different neural network models specifically for segmentation to see which one works best for this type of task._

_Third, we’ll evaluate the models’ performance using metrics like Intersection over Union, or IoU. Finally, the best-performing model will be deployed in a real-time monitoring system for continuous tracking of spills._

_Each objective brings clear benefits: increased accuracy with machine learning, greater efficiency in detection, valuable contributions to environmental research, and enhanced monitoring capabilities. Altogether, these benefits make our project a powerful tool for protecting oceans and responding to oil spills quickly._

--- 

## Bianca 

### Slide 4-7: Stakeholders (4 slides, 30 seconds each)

We have identified key stakeholders who are directly impacted by oil spill detection technology.

- **Slide 4**: Government and environmental agencies are vital, as they play a major role in managing natural resources and enforcing environmental regulations. They benefit directly from efficient spill detection in their conservation efforts.
  
- **Slide 5**: Academic research institutions are crucial contributors, bringing expertise in environmental science and advanced technology. Their research drives innovation in oil spill detection, helping to improve accuracy and effectiveness. By collaborating on this project, these institutions benefit from real-world applications of their research, enhancing their contributions to environmental conservation and technological advancement.

- **Slide 6**: Data providers and competitors are essential in supplying the raw data and fostering a competitive environment that drives progress. Organizations like satellite imaging companies and research labs enable timely and precise detection, while competition pushes continuous improvement in methods.

- **Slide 7**: Project sponsors and partners provide the funding and strategic guidance needed to turn research into real-world solutions. Their support ensures the project’s viability and long-term impact. In return, they gain recognition for contributing to a critical environmental cause, as well as potential commercial applications and opportunities for innovation within their own sectors.

### Slide 8: Interest x Influence Matrix (30 seconds)

To better understand stakeholder dynamics, we developed an interest versus influence matrix. This matrix helps us prioritize communication and strategy efforts by mapping stakeholders according to their level of interest in oil spill detection and their influence over decision-making processes. High-interest, high-influence stakeholders, such as government agencies and the oil industry, require ongoing engagement and collaboration.

### Slide 9: Functional Diagram (30 seconds)

Let’s look at the functional diagram of our system. We start with SAR images, which go through data preprocessing to enhance the input quality. These images are then fed into our implemented models, which include U-Net, LinkNet, PSPNet, DeepLab, and CBDNet. Each model processes the input, generating segmented output images where potential oil spills are highlighted in distinct colors. The performance of these models is evaluated using different evaluation accuracy metrics like Intersection over Union. The processed data and results are then shared with our primary stakeholders: Public Authorities and Regulatory Bodies, Academic Research Institutions, and the European Space Agency. This system not only automates spill detection but provides valuable insights for stakeholders involved in environmental conservation and research.

---
## Ale

### Slide 9: User-Personas: Maria (20 seconds)

We managed to identify three typical user personas that embody the typical stakeholder of our project. Meet Maria, an Environmental Regulator representing public authorities. Maria faces challenges with delayed or inaccurate data, which affects her ability to respond swiftly to oil spill incidents. Her primary goal is to have access to accurate, real-time oil spill detection tools to support timely decision-making and reduce environmental damage. Additionally, she values data visualization tools that can translate complex satellite data into easily interpretable formats for non-technical stakeholders, enabling broader understanding and faster action.

### Slide 9: User-Personas: Adam (20 seconds)

Here we have Adam, a Marine Research Scientist whose work focuses on oil spill detection. Adam faces challenges with limited access to high-quality, labeled SAR data for training and validating neural network models. Additionally, data preprocessing and integration are time-consuming tasks that hinder his research progress. Adam's primary goal is to access a reliable data source and a robust detection model that can support his research, enabling him to advance oil spill detection techniques effectively.

### Slide 10: User-Personas: Thomas (20 seconds)

Introducing our last user persona, Thomas, an ESA Project Manager responsible for ensuring data accessibility and reliability, particularly for high-priority projects like oil spill detection. Thomas faces significant pressure to maintain efficient SAR data processing and distribution channels. His goals include establishing partnerships to increase the usage of satellite data and supporting environmental protection initiatives. Reliable data flow and strategic collaborations are essential for Thomas to meet his project objectives and promote impactful environmental monitoring.

### Slide 11: User Stories (10 seconds)

This slide presents the user stories for each of our key personas. These stories capture the specific needs, goals, and motivations of our users, providing clear direction for how the system should be designed to meet their requirements effectively.

### Slide 12: WBS (30 seconds)

Now we move into the management section, starting with the Work Breakdown Structure, or WBS. This breakdown provides an overview of the project’s key work packages, detailing each phase from Project Management to Documentation. Each work package is assigned a lead, with estimated project management effort in person-months, and specific start and end months. This structure helps us allocate resources, set timelines, and track responsibilities, ensuring that each stage of the project is organized and progresses smoothly.

### Slide 13-14: GANTT (30 seconds)

These slides present the Gantt chart for our project, organized around six main work packages: Project Management, Research & Familiarization, Data Exploration & Preprocessing, Model Implementation & Training, Evaluation & Comparison, and Documentation & Communication. The Gantt chart visually maps out the timeline and progress of each task within these packages, showing assigned team members, start and end dates, and completion status. This structured approach allows us to manage our workflow, ensure tasks are on schedule, and stay aligned with our project milestones.