# E-Commerce Models

Multiple projects as part of an 'Electronic Commerce' course, covering topics such as network dynamics, MABs and recommendation systems.

The credit for the project descriptions (originally in Hebrew) and template code goes to the staff of the 'Electronic Commerce' course at Technion, spring semester of 2023. The solutions are our own work.

## 1. [Network Dynamics](1_Influence_in_Networks)

### Background
As excellent students in the E-commerce Models course, you have been approached by DollsRUs, a large toy store, to help them promote their new doll collection on the social media platform "Chic Choc". Chic Choc allows users to upload videos, particularly showing the dolls. When a user buys a doll from DollsRUs's new collection, their friends on the network can see this purchase.

### Task
You need to select 5 influencers to receive a free doll from the new collection at time t=0. When a user receives or buys a doll, their friends on the network see the purchase and decide with a certain probability whether to buy the doll as well. Your goal is to maximize the number of users who buy the dolls over 6 time periods (i.e., at times t=1,2,…,6).

### Technical Notes

The probability of a user buying a doll depends on their friends on Chic Choc.
The network structure is dynamic, and the probability of a link forming between two users is influenced by their common neighbors.
The file chic_choc_data.csv describes the network at  t=0.
The cost of hiring each influencer is provided in the file costs.csv.

### Objective
Maximize the company's profit from sales, subtracting the cost of hiring influencers. Formally, maximize E(R)−C, where E(R) is the expected number of purchasers and C is the cost of influencers.

### Our Solution

We found the best nodes using a combination of greedy (hill climb) and random search algorithms. We then used a Monte Carlo simulation to estimate the expected number of purchasers for each set of influencers. Finally, we calculated the expected profit for each set of influencers and selected the one with the highest profit.

We also tried other approaches, notably graph properties and improving upon a chosen set.

## 2. [Multi-Armed Bandits](2_Multi-Armed_Bandits)

### Background
After your success in the first project with DollsRUs, you have now been approached by "Flexnit" to develop a recommendation algorithm for their system. The recommendation system involves three types of participants:

- Content Consumers: Users of the system, divided into groups with different preferences.
- Content Producers: Creators of the content.
- Commercial Entity: Flexnit, which operates the recommendation system.

### Task
Develop an algorithm that recommends TV shows to users of the Flexnit platform. After receiving a recommendation, the user watches the show and reports their satisfaction. The user's satisfaction (or reward) from content producer j is a random variable from an unknown distribution. The goal of the algorithm is to learn these distributions to make the best recommendations.

Content producers invest money to display their series on Flexnit, and they earn a dollar each time a user watches their content. If a content producer doesn't get enough views, they cancel their contract with Flexnit, making their series unavailable. Recommending such a series results in zero reward for the user. Each content producer has a threshold for the number of required views over a specific time window.

### Simulation Workflow
Each simulation runs for a set number of rounds. In each round, a user arrives at the system, and your algorithm recommends a content producer. The recommendation results in a reward, which is a random variable. Your goal is to maximize the total rewards. The system checks which content producers are satisfied at the end of each time window and marks inactive ones. The reward from inactive producers is zero.

### Our Solution

The solution involves several steps. First, we implemented the Contextual Multi-Armed Bandits (CMAB) algorithm using Upper Confidence Bound (UCB) to estimate user rewards sampled from a uniform distribution. Initially, each user tries every arm to establish a sample, and we update our estimates and counts after each round. 

In the second step, we focused on preventing arm deactivation by allowing the algorithm to choose the best arm for the user and intervening only when an arm is about to be deactivated. This ensures all arms meet their thresholds. 

In the third step, we used mini-simulations to decide which arms to allow for deactivation since they have a low benefit, running multiple scenarios for each subset of arms and choosing the best one based on estimated rewards. We applied an explore-exploit principle to select the best subset towards the end of the explore phase. 

The algorithm achieves a reward of approximately 514,000 in simulations and handles various scenarios effectively.

## 3. [Recommendation Systems](3_Recommendation_Systems)

### Background
Following your previous successes, you have been approached by "Banana" to help develop a new preference prediction system for their music app, Banana Music. The app recommends songs to users based on their listening history. You are provided with data on how often users listened to specific songs.

### Task
Use the provided listening data to predict how many times each user will listen to given songs listed in test.csv. The number of listens indicates user preference, functioning as a rating.

The goal is to minimize the error function, which is the sum of squared differences between the actual number of listens and the predicted number of listens:
$$ f = \sum_{(u,s) \in \text{train}} (r_{u,s} - \hat{r}_{u,s})^2$$

There are four methods to implement:

- **Task 1:** Predict the number of listens using the baseline model: 
$\hat{r}_{u,s} = r_{avg} + b_u + b_s$.

- **Task 2:** Implement the alternating least squares method to predict listens using feature vectors $ \hat{r}_{u,s} = p_u^T\cdot q_s$.

- **Task 3:** Use SVD for low-rank approximation with sk=20 to predict listens.

- **Task 4:** An open-ended, competitive task where you can use any method to minimize 

### Our Solution

1. Reward = 932,156,017,095.7792. 
We created matrix A with our data and vector r containing the true values. We found the bias values for each user and song by solving the least squares equation corresponding to the given problem.

2. Reward = 298,116,335.2026184. We created matrices for all users and all songs. Using the principle of alternating least squares, we computed the best feature vectors for each user and song to best describe them.

3. Reward = 229,084,139,946.93808. We created a sparse matrix from the given data and applied SVD with a parameter of 20 to get a low-rank approximation. To predict the number of listens for any user-song pair, we multiplied the resulting matrices at the appropriate indices to get the approximation.

4. Reward = 999,285,762,909.2153. We split the labeled data into training and test sets and experimented with various hyperparameters for Tasks 2 and 3, evaluating results on the test set. We added regularization to Task 2, improving predictions. We then tried an ensemble approach, averaging the weighted predictions of the three methods based on their previous success. This approach achieved results around 23 × 10^10. We also experimented with other methods, like finding similar users/songs using different distance functions, but these were less effective and had higher runtime. Ultimately, we found the best predictions using the naive averaging method from the lecture, adding biases for users and songs computed using the first averaging method. This gave the best results on our test set, achieving test reward of 147,589,413,768.45416.
