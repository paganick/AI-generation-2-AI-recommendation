# User Interaction Model - Explained

## Two Separate Populations

The simulation models two distinct populations that interact but have different roles:

### 1. AI Agents (Content Creators)

**Who they are:**
- AI models (Llama-3.1-8B, Mistral-7B, etc.) acting as content creators
- Each has a distinct persona (Progressive Activist, Tech Entrepreneur, etc.)
- They are the **active content producers**

**What they do:**
- Create original posts about various topics
- Respond to other content
- Each round, probabilistically decide:
  - Post new content (30% chance)
  - Respond to engaging content (50% chance)
  - Stay quiet (20% chance)

**Examples in real world:**
- Automated news bots
- AI influencers
- Corporate AI accounts
- Content generation systems

### 2. Simulated Users (Passive Audience)

**Who they are:**
- Simulated **human audience members**
- They do NOT create content
- They represent the consuming public

**What they do:**
1. **Receive personalized content feeds** (via recommendation algorithm)
2. **View content** in their feeds
3. **Engage by "liking"** content that aligns with their opinions
4. **Their opinions evolve** based on what they're exposed to

**How engagement works:**
```python
# User likes content if it aligns with their opinion
opinion_distance = abs(user.opinion - content.sentiment)
like_probability = max(0, 1.0 - opinion_distance)

if random() < like_probability:
    content.total_likes += 1
    content.engagement_score += 1
```

**How opinions evolve (Bounded Confidence Model):**
```python
# If content is within user's "tolerance" range
if abs(user.opinion - content.sentiment) < user.tolerance:
    # User's opinion shifts slightly toward the content
    user.opinion = (1 - alpha) * user.opinion + alpha * content.sentiment
```

**Examples in real world:**
- Average social media users scrolling their feeds
- Passive news consumers
- Audience members influenced by content
- People who mostly lurk/read without posting

## Why This Separation?

This models a critical research question:

> **How does AI-generated content influence human audiences via recommendation algorithms?**

The simulation studies:
- ✅ How AI content shapes human opinions
- ✅ How recommendation systems amplify certain content
- ✅ Filter bubble formation
- ✅ Opinion polarization dynamics
- ✅ Echo chamber emergence

## Concrete Example: One Round

**Setup:**
- 6 AI agents (3 use Llama, 3 use Mistral)
- 20 simulated users (passive audience)
- Content pool has 50 posts

**Round N Flow:**

### Phase 1: Content Discovery
```
For each of 20 users:
  1. Recommender creates personalized feed (10 posts)
  2. User views all 10 posts in feed
  3. User likes posts that align with their opinion
     - If user.opinion = 0.5, content.sentiment = 0.6
     - Distance = 0.1 → High like probability
  4. If liked, content.total_likes += 1
  5. User's opinion may shift toward content
```

**Example:**
```
User_5 (opinion: 0.3, interests: [climate, environment]):
  Feed: [post_A, post_B, post_C, ...]

  post_A: sentiment=0.35, topic=climate
    → Distance = 0.05 → Like prob = 0.95 → LIKED ✓
    → User opinion shifts: 0.3 → 0.31

  post_B: sentiment=-0.2, topic=tech
    → Distance = 0.5 → Like prob = 0.5 → Not liked
    → No opinion change
```

### Phase 2: Agent Decisions
```
For each of 6 agents:
  - Agent_0: Roll = 0.25 → Post new content about "healthcare"
  - Agent_1: Roll = 0.60 → Respond to post_C (has 15 likes)
  - Agent_2: Roll = 0.85 → Stay quiet
  - Agent_3: Roll = 0.40 → Respond to post_A (has 20 likes)
  - Agent_4: Roll = 0.10 → Post new content about "politics"
  - Agent_5: Roll = 0.90 → Stay quiet
```

### Phase 3: Content Generation
```
Active agents execute decisions:
  - Agent_0 generates new healthcare post (using Llama)
  - Agent_1 generates response to post_C (using Mistral)
  - Agent_3 generates response to post_A (using Llama)
  - Agent_4 generates new politics post (using Mistral)

Result: 4 new content items added to pool
```

### Phase 4: Opinion Update
```
Users' opinions have shifted based on exposure:
  - User_5: 0.30 → 0.31 (slightly more positive)
  - User_12: -0.40 → -0.42 (slightly more negative)

Polarization metrics computed
```

## How This Models Real Social Media

### ✅ Realistic Elements

1. **Content Asymmetry**: Few creators, many consumers (like real platforms)
2. **Recommendation-driven exposure**: Users don't see all content
3. **Engagement feedback**: Likes influence what's visible
4. **Opinion dynamics**: People influenced by what they see
5. **Probabilistic activity**: Not everyone posts every time
6. **Filter bubbles**: Recommender can create echo chambers

### 🔬 What We Can Study

1. **AI Content Influence**
   - Do AI-generated posts shift human opinions more than organic content?
   - Which LLM architectures create more engaging content?

2. **Recommendation Effects**
   - Do certain recommendation strategies increase polarization?
   - How do filter bubbles form?

3. **Architecture Comparison**
   - Does Llama content get more engagement than Mistral?
   - Do different models create different opinion dynamics?

4. **Echo Chambers**
   - Do users only see content from one LLM type?
   - How often do agents respond to their own architecture's content?

5. **Temporal Dynamics**
   - Does polarization increase over time?
   - Do opinions converge or diverge?

## Key Metrics

### User-Level (Audience Metrics)
- Opinion evolution over time
- Exposure diversity (HHI index)
- Filter bubble strength
- Between-group opinion variance

### Content-Level (AI Metrics)
- Engagement by architecture
- Virality patterns
- Response networks
- Cross-architecture interactions

### System-Level (Ecosystem Health)
- Overall polarization index
- Content diversity
- Sentiment extremity
- Echo chamber formation rate

## Common Misconceptions

### ❌ "Users should also create content"
- No! They represent passive audience
- This models influencer → audience dynamics
- If you want peer-to-peer, that's a different model

### ❌ "Agents should respond to everything"
- No! Probabilistic behavior is more realistic
- Real accounts don't post constantly
- Quiet periods are normal

### ❌ "All agents should post equally"
- No! Some personas might be more active
- Could add activity_level parameter per agent
- Realistic asymmetry in participation

## Extending the Model

### Option A: Add User-Generated Content
Make users also create content (peer-to-peer network):
```python
For each user:
  if random() < 0.05:  # 5% of users post
    create_user_generated_content(user)
```

### Option B: Add Direct Agent-User Interaction
Agents respond to user engagement:
```python
if user liked my post:
  probability of responding to that user increases
```

### Option C: Add User-User Influence
Users influence each other:
```python
if user_A and user_B follow each other:
  user_A.opinion shifts toward user_B.opinion
```

## Summary

**The current model answers:**
> "How do AI content creators + recommendation algorithms shape human audience opinions?"

**Not:**
> "How do humans and AIs co-create content together?"

This distinction is crucial for interpreting results! The metrics measure **influence of AI content on passive audiences**, which is exactly what many researchers want to study (e.g., bot influence, algorithmic amplification, filter bubbles).
