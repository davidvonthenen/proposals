## Embracing Change and Formulating Strategies for Dev/Test

**Proposal for OSCON US 2016**  

**Description:**  
Using our recent release of an Apache Mesos Isolator project as an example, we will explore strategies for leveraging technologies such as Docker and AWS to enable quicker, efficient and collaborative development and test.

**Topic:**  
Distributed Development Stack

**Secondary Topics:**  
Techniques, Tools

**Session Type:**  
40 minute presentation

**Abstract (Longer, more detailed description (3-6 paragraph, bullet points welcome) of your presentation to help the program committee understand what you will cover. If your proposal is chosen, this is the description that will appear on the website. Note that our copywriters may edit it for consistency and O'Reilly voice.):**  
Coming from a company that is viewed for developing enterprise software by traditional methods, our group has been chartered to not only live the Open Source life, but also to create software by following in the footsteps of startups. That means not only do as startups do by embracing new technologies, but by doing so with transparency, open discussion, and contribution with the community.

This session will cover lessons learned about embracing Open Source and thoughts on development and test using a variety of technologies including but not limited to Docker, AWS, and etc.  
1. Use the Tools
  - GitHub
  - Slack
  - Google Hangouts
  - Yes, even you cell phone
2. Dev
  - Docker Build Environment Images
    - Build build images (sounds redundant but not)
    - Component integration early on
    -	Versioning via tags
    -	(Aspire to have a) Quick Demo - mesos-dev with Docker tags
      - Run “docker images”
  - Accelerate Building Docker Images in AWS
    - Build images using AWS
    - Spin up, spin down
    - (Aspire to have a) Quick Demo - Kick off Docker Image Build
      - In AWS, start but not finish a Docker build
  - Dogfooding
    - Docker images to build your own project(s)
    - Saves others from standing up environments to build your project
    - (Aspire to have a) Quick Demo - Kick off Docker Image Build
      - Kick off the Mesos Isolator project build (run in a couple of minutes)
3. Test
  - Docker Test Environments
    - Test baselines with qualified/known versions of components
    - Avoids surprises
    - Concurrency – Test Images can be built concurrent to Build Images
    - (Aspire to have a) Quick Demo – Create Docker Test Image
  - In the Cloud
    - Stand up Test in AWS
    - Promotes collaboration for debugging
    - Demos and Presentations
    - (Aspire to have a) Quick Demo – Start standup of Test env
      - Kick off a deployment of (but not finish) a Test configuration
  - CI
    - Remember those build images?
    - Regression testing through CI (Travis on GitHub)
  - Reuse
    - Dev can use “blessed” Test Images to harden code
    - Versioning for revisit for user issues
4. Publish, Publish, Publish
  - Publish to Docker Hub
  - Easier for others to contribute by pulling Docker images
    - Build Image for your Project
    - Test Images for Verification
    - (Aspire to have a) Quick Demo – Show Docker Hub page
      - Open up my Docker Hub page with my available images
  - Software Lifecycle
    - Issues – re-spin up a build and test environment for quicker turn around


**Additional Tags:**  
Docker, AWS, Development, Test, Mesos, Apache

**What’s the takeaway for the audience:**  
Take the ideas and lessons learned from our team and formulate strategies on how you can grow, supplement or even step away from traditional Dev/Test environments, but more importantly how these concepts promote collaboration and efficient within your organization.

**Audience Level:**  
Intermediate

**Prerequisite:**  
First hand software development experience

**Conceptual or How-To:**  
How-to

**Tutorial hardware and/or Installation Requirements:**  
NA

**Video URL:**  
NA

**O’Reill Author:**  
NA

**Recommend or encourage you to submit a proposal:**  
Clinton Kitson <clinton.kitson@emc.com>

**Diversity:**  
No

**Travel & Other Expense:**  
No

**If so, please describe:**  
NA

**Additional Notes:**  
NA
