### Embracing Change and Formulating Strategies for DevTest

**Proposal for DockerCon US 2016**

**First Name:**  
David

**Last Name:**  
vonThenen

**Email Address:**  
david.vonthenen {at} emc {dot} com

**Title:**  
Developer Advocate

**Company:**  
EMC {code}

**Personal Blog:**  
http://dvonthenen.com

**List the previous industry conferences you've spoken at?:**  
EMC World

**Link to videos from previous talks:**  
NA

**Twitter Handle**  
dvonthenen

**Speaker bio (75 words max):**  
David vonThenen is a Developer Advocate at EMC. He is currently a member of the EMC {code} team which lives and breathes Open Source by making contributions to the community in a wide variety of projects ranging from Apache Mesos to storage orchestration platforms. Prior to joining EMC {code}, David was a technical architect and development lead for a Backup/Recovery solution with heavy focus in the virtualization space, VMware in particular. David has previous experience in areas ranging in semiconductors, mainframe computing, and iSCSI/FC/FCoE storage initiators and targets.

**What's your relationship to Docker ?:**  
Contributor

**Session Title:**  
Embracing Change and Formulating Strategies for Dev/Test

**Abstract:**  
The way software is being developed is changing rapidly. The need for in-house labs with expensive power and cooling systems filled with rack of underutilized servers are almost behind us. Many of these startups are getting by on the cheap not because VC money isn't there, but rather because those legacy modes of operation are not agile enough or offer very little towards collaboration. This session will cover lessons learned about embracing Open Source and thoughts on software development and test using a variety of technologies including but not limited to Docker, AWS, and etc.

**Outline:**  
Proposal Outline:
- How the Tools Have Changed
  - Communication
    - GitHub
    - Slack
    - Google Hangouts
    - Yes, even you cell phone
  - IDEs
    - Atom
    - JHipster
    - Komodo
    - Cloud9
  - Infrastructure
    - Docker
    - VirtualBox
    - AWS
    - GCE
- Development
  - Docker Build Environment Images
    - Build build images (sounds redundant but not)
    - Component integration early on
    -	Versioning via tags
    -	Quick Demo - mesos-dev with Docker tags
      - Run “docker images”
  - Accelerate Building Docker Images in AWS
    - Build images using AWS
    - Spin up, spin down
    - Quick Demo - Kick off Docker Image Build
      - In AWS, start but not finish a Docker build
  - Dogfooding
    - Docker images to build your own project(s)
    - Saves others from standing up environments to build your project
    - Quick Demo - Kick off Docker Image Build
      - Kick off the Mesos Isolator project build
- Testing
  - Using Docker
    - Test baselines with qualified/known versions of components
    - Avoids surprises
    - Quick Demo
      - Testing using Minimesos (a Docker backed environment)
  - In the Cloud
    - Testing in AWS
    - Promotes collaboration for debugging
    - Demos and Presentations
  - CI
    - Remember those build images?
    - Regression testing through CI (TravisCI on GitHub)
- Publish, Publish, Publish
  - Publish to Docker Hub
  - Easier for others to contribute by pulling Docker images
    - Build Image for your Project
    - Test Images for Verification
    - Patches or Hotfixes
      - Re-spin up a build and test environment for quicker turn around
    - Quick Demo – Show Docker Hub page
      - Open up my Docker Hub page with my available images
      - Pull a couple of Docker images

**What are the key takeaways from your session? :**  
- Introduce other ways of thinking about and using Docker
- Formulate strategies on how you can grow, supplement or even step away from traditional Dev/Test environments
- All components used for the demo is available on GitHub

**Are you able to share details about your application architecture/design and implementation results like metrics?:**  
Yes

**Keywords**  
Development, Testing, DevTest, Docker, Docker Engine, Docker Hub, AWS, MiniMesos, AWS

**Can your CFP be turned into a lightning talk?:**  
No

**What theme do you believe your talk fits into?:**  
Culture

**Expertise level:**  
Intermediate

**Who's the main target audience?:**  
Developer or Development Manager

**Is your talk related to any of the following Docker Projects?:**  
Docker Engine
Docker Hub

**Does your presentation have the participation of a woman, person of color, person with disabilities, or member of another group often underrepresented at tech conferences?:**  
No

**Will you have a co-presenter?:**  
No

**Agree to the code of conduct:**  
Yes
