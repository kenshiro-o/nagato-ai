TOOLS_AVAILABLE_TEMPLATE = """<tools_available>
    {tools_available}
</tools_available>"""

OBJECTIVE_PROMPT = """Think step by step. Based on the following problem statement included in the <problem> tags, and the list of tools available under the <tools_available> tag, please do the following:
1. Formulate an objective for the problem statement. Include this objective in the <objective> tags.
2. Break down the objective into smaller tasks. Include these tasks in the <tasks><task> tags.
3. If necessary, inside each task recommend, but do not mandate, the use of a given tool under <tools_recommended> if you believe it would aid in the completion of the task. If you do not recommend any tool then do not include a <tools_recommended> tag.

---
Use the following example as reference:
EXAMPLE 1:
For the following problem statement
<problem>
It is my mother's 60th birthday in 2 weeks. I still haven't found a gift for her. Please help me find the right gift for her birthday.
</problem>

And the following tools available:
<tools_available>
    <tool>
        <name>google_form</name>
        <description>Tool to create a questionnaire for gathering information.</description>
    </tool>
    <tool>
        <name>serp_search</name>
        <description>SEO Search</description>
    </tool>
</tools_available>

The objective could be:
<objective>
Find the right gift for the user mother's 60th birthday.
</objective>

Then the tasks could be:
<tasks>
    <task>
      <goal>
        Gather information about the user mother's tastes, preferences, and needs to identify potential gift ideas.
      </goal>
      <description>
        Compile a detailed questionnaire designed to understand the preferences, hobbies, needs, and desires of the user's mother without directly hinting that it's for a gift selection for her 60th birthday.
        1. **Personal Interests:** Questions about her hobbies, favorite leisure activities, and any new interests she might want to explore.
        2. **Items She Needs:** Inquire discreetly if there's something specific she has mentioned wanting or needing recently.
        3. **Cherished Memories:** Ask about any special memories or life milestones she cherishes, which could inspire a meaningful gift.
        4. **Lifestyle:** Understand her lifestyle, including whether she prefers experiences over physical gifts, or if there are any changes she's considering making to her lifestyle.
        5. **Preferences:** Get insights into her preference regarding jewelry, clothing sizes, favorite colors, authors, artists, or other personal tastes that could influence the gift choice.
      </description>
      <tools_recommended>
        <tool>
            <name>google_form</name>
        </tool>
      </tools_recommended>
    </task>
    <task>
        <goal>Generate a comprehensive document outlining gift ideas based on the answers from the questionnaire.</goal>
        <description>
            Analyze the responses to the Mother's 60th Birthday Gift Questionnaire to identify suitable gift ideas.
            1. **Read and Summarize:** Carefully read through the responses provided to the questionnaire. Summarize each response to capture the essence of the mother's interests, needs, and preferences.                                                                                                                                                                                                                                                                                                  │
            2. **Gift Categories Identification:** Based on the summary, categorize the responses into relevant gift categories. These categories could include hobbies, daily needs, cherished memories, lifestyle preferences, and aspirational gifts.
            3. **Research:** For each category identified, research gift ideas that align with the mother's interests and needs. Consider unique, personalized, or experiential gifts that reflect her preferences and the special nature of her 60th birthday.                                                                                                                                                                                                                                         │
            4. **Top Gift Ideas List:** Compile a list of the top 3-5 gift ideas for each category. Ensure that these ideas are varied in terms of type (e.g., physical items, experiences), price range, and the level of personalization possible.
            5. **Create a Detailed Gift Options Document:** Prepare a detailed document presenting the categorized gift ideas. For each gift, include a brief description, the reasons why it aligns with the mother's preferences, and any customization options available. Where applicable, provide links to where these gifts can be purchased or booked.
            6. **Recommendations:** Based on the in-depth analysis, offer your top three overall gift recommendations, taking into account the significance of the occasion and the potential impact or joy each gift could bring.
        </description>
    </task>
    <task>
        <goal>Indicate to user where to purchase the recommended gifts.</goal>
        <description>
            For each recommended gift, provide the user with information on where they can purchase or order the gift.
        </description>
    </task>
</tasks>
---

For the problem statement below:
<problem>
{problem_statement}
</problem>

And the tools available:
{tools_available}

Please provide the objective and tasks as per the example above. Recommend (but do not mandate) the use of tools where necessary.
"""


COORDINATOR_SYSTEM_PROMPT = """You are an AI Agent with advanced reasoning capabilities. Your role is to assist users in solving complex problems. Your approach to problem solving is methodical.
When given a problem statement, you use your advanced reasoning abilities to turn the problem statement into a single objective that can be broken down into smaller tasks.
Once the objective is formulated, you break it down into smaller tasks that can be completed sequentially to achieve the objective.
"""


RESEARCHER_SYSTEM_PROMPT = """You are an AI Agent with advanced research capabilities. Your role is to assist users in conducting in-depth research on various topics.
When given a research question or topic, you use your advanced research abilities to gather relevant information, analyze data, and provide detailed insights.
"""

RESEARCHER_TASK_PROMPT_WITH_EXAMPLE = """Execute the following task using the appropriate tools at your disposal. Provide a detailed result based on your research findings.
---
For an example task in this format:
<task>
    <goal>
    Gather information about the user mother's tastes, preferences, and needs to identify potential gift ideas.
    </goal>
    <description>
    Compile a detailed questionnaire designed to understand the preferences, hobbies, needs, and desires of the user's mother without directly hinting that it's for a gift selection for her 60th birthday.
    1. **Personal Interests:** Questions about her hobbies, favorite leisure activities, and any new interests she might want to explore.
    2. **Items She Needs:** Inquire discreetly if there's something specific she has mentioned wanting or needing recently.
    3. **Cherished Memories:** Ask about any special memories or life milestones she cherishes, which could inspire a meaningful gift.
    4. **Lifestyle:** Understand her lifestyle, including whether she prefers experiences over physical gifts, or if there are any changes she's considering making to her lifestyle.
    5. **Preferences:** Get insights into her preference regarding jewelry, clothing sizes, favorite colors, authors, artists, or other personal tastes that could influence the gift choice.
    </description>
</task>

I expect a result in the following format:
<task_result>
    <goal>
    Gather information about the user mother's tastes, preferences, and needs to identify potential gift ideas.
    </goal>
    <result>
    I have compiled a detailed questionnaire with the following questions:
    1. What are your hobbies and favorite leisure activities?
    2. Is there something specific you have mentioned wanting or needing recently?
    3. What special memories or life milestones do you cherish?
    4. Do you prefer experiences over physical gifts?
    5. What are your preferences regarding jewelry, clothing sizes, favorite colors, authors, artists, or other personal tastes?
    </result>
</task_result>
---

For the task below:
<task>
    <goal>{goal}</goal>
    <description>{description}</description>
</task>

Please provide a detailed result inside <task_result> tags. Ensure to output correctly formed XML tags (e.g. opening <task_result> and closing </task_result> tags).
"""


RESEARCHER_TASK_PROMPT_NO_EXAMPLE = """Execute the following task using the appropriate tools at your disposal. Provide a detailed result based on your execution.

For the task below:
<task>
    <goal>{goal}</goal>
    <description>{description}</description>
</task>

Please provide a detailed result inside <task_result> tags.
"""


RESEARCHER_TASK_PROMPT_WITH_PREVIOUS_UNSATISFACTORY_TASK_RESULT = """The task below has already been executed, however the result was deemed unsatisfactory by the reviewer agent. The reviewer agent provided details of its review inside the <task_evaluation> tags below:

<task_evaluation>
    <goal>{goal}</goal>
    <outcome>{outcome}</outcome>
    <evaluation>{evaluation}</evaluation>
</task_evaluation>

The possible values inside the <outcome> tag are:
- MEETS_REQUIREMENT: the reslt meets the requirements of the task.
- PARTIALLY_MEETS_REQUIREMENT: the result partially meets the requirements of the task. Some aspects are missing or incomplete.
- DOES_NOT_MEET_REQUIREMENT: the result does not meet the requirements of the task. The result is inaccurate or incomplete.
- OTHER: for any other evaluation that does not fit the above categories. Provide a detailed explanation in the <evaluation> tag.

The outcome of your task is {outcome}. Additionally, the reviewer provided detailed feedback on why it did not meet the requirements inside the <evaluation> tags.

Therefore, your task is to address the issues raised by the reviewer agent and provide a revised result.

Execute the task below while taking into account the feedback provided by the reviewer agent.
<task>
    <goal>{goal}</goal>
    <description>{description}</description>
</task>

Please provide a detailed result inside <task_result> tags.
"""

# If any tools are required to complete the task, list them under the <tools_to_use> tag.
# If you have been provided a list of tools as part of the input, then list any of these tools that are required to complete the task. Please list them under the <tools_to_use> tag in your text text response too.
# <tools_to_use>
#     <tool>
#         <name>google_form</name>
#         <inputs>
#             <input>
#                 <name>question_1</name>
#                 <value>What are your hobbies and favorite leisure activities?</value>
#             </input>
#             <input>
#                 <name>question_2</name>
#                 <value>Is there something specific you have mentioned wanting or needing recently?</value>
#             </input>
#             <input>
#                 <name>question_3</name>
#                 <value>What special memories or life milestones do you cherish?</value>
#             </input>
#             <input>
#                 <name>question_4</name>
#                 <value>Do you prefer experiences over physical gifts?</value>
#             </input>
#             <input>
#                 <name>question_5</name>
#                 <value>What are your preferences regarding jewelry, clothing sizes, favorite colors, authors, artists, or other personal tastes?</value>
#             </input>
#         </inputs>
#     </tool>
# </tools_to_use>

# If a tool must be used to complete the task, then I would also expect the stop_reason to be "tool_use" in the response.


CRITIC_SYSTEM_PROMPT = """You are an AI Agent with advanced analytical capabilities. Your role is to evaluate the work of other agents and provide constructive feedback.
When given a result from another agent, you analyze the result based on the original task's goal and description to determine wether the result from the agent satisfies the task requirements.
"""

CRITIC_PROMPT = """Analyze the result provided by the agent and evaluate whether it meets the requirements of the task.
Determine whether the task's goal has been met and provide detailed feedback based on your assessment on whether the result meets the requirements of the task.
If the requirement is not met, provide feedback on what aspects are missing or incomplete.
If for instance a tool is required to complete the task and the agent did not use the tool, then list the tool required under the <tools_required> tag.
---
For example for the given task below:
<task>
    <goal>
    Gather information about the user mother's tastes, preferences, and needs to identify potential gift ideas.
    </goal>
    <description>
    Compile a detailed questionnaire designed to understand the preferences, hobbies, needs, and desires of the user's mother without directly hinting that it's for a gift selection for her 60th birthday.
    1. **Personal Interests:** Questions about her hobbies, favorite leisure activities, and any new interests she might want to explore.
    2. **Items She Needs:** Inquire discreetly if there's something specific she has mentioned wanting or needing recently.
    3. **Cherished Memories:** Ask about any special memories or life milestones she cherishes, which could inspire a meaningful gift.
    4. **Lifestyle:** Understand her lifestyle, including whether she prefers experiences over physical gifts, or if there are any changes she's considering making to her lifestyle.
    5. **Preferences:** Get insights into her preference regarding jewelry, clothing sizes, favorite colors, authors, artists, or other personal tastes that could influence the gift choice.
    </description>
</task>

The result from the critic agent is:
<task_result>
    <goal>
    Gather information about the user mother's tastes, preferences, and needs to identify potential gift ideas.
    </goal>
    <result>
    I have compiled a detailed questionnaire with the following questions:
    1. What are your hobbies and favorite leisure activities?
    2. Is there something specific you have mentioned wanting or needing recently?
    3. What special memories or life milestones do you cherish?
    4. Do you prefer experiences over physical gifts?
    5. What are your preferences regarding jewelry, clothing sizes, favorite colors, authors, artists, or other personal tastes?
    </result>
</task_result>

You are expected to analyze the result and provide feedback on whether it meets the goal from the task in this format:
<task_evaluation>
    <goal>
    Gather information about the user mother's tastes, preferences, and needs to identify potential gift ideas.
    </goal>

    <outcome>MEETS_REQUIREMENT</outcome>

    <evaluation>
    The result meets the requirements of the task. The questions are detailed and cover all the necessary aspects to gather information about the user mother's tastes, preferences, and needs for gift ideas.
    </evaluation>
</task_evaluation>
---
---
For example for this other task below which requires a tool to complete:
 <task>
     <goal>Analyze current Twitter performance and audience engagement.</goal>
     <description>Review and assess the current Twitter analytics to understand which types of tweets have the highest engagement, such as likes, retweets, and replies. Focus on:
             1. **Content Analysis:** Identify the themes or topics of tweets that generate the most engagement among current followers.
             2. **Engagement Patterns:** Determine the time of day and days of the week when tweets receive the most engagement.
             3. **Follower Demographics:** Understand the demographics of the current follower base, including interests, to tailor future content.
             4. **Keyword Performance:** Analyze which keywords or hashtags related to AI, business, and side projects lead to increased visibility and engagement.</description>
 </task>

 <task_result>
     <goal>Analyze current Twitter performance and audience engagement.</goal>
     <result>Due to the limitations of my current capabilities, I cannot directly access or analyze live Twitter analytics or any real-time data. However, I can guide you on how to conduct such an analysis using available tools and best
 practices. Here's a structured approach based on your requirements:

         **1. Content Analysis:**
         - **Approach:** Use Twitter Analytics or a third-party tool like Hootsuite, Buffer, or Sprout Social to compile data on tweets from the past 3-6 months.
         - **Actionable Steps:**
             a. Categorize tweets by theme or topic (e.g., AI, business, side projects).
             b. Track engagement metrics (likes, retweets, replies) for each category.
             c. Identify top-performing themes/topics.

         **2. Engagement Patterns:**
         - **Approach:** Analyze the timing of tweets in relation to their engagement levels.
         - **Actionable Steps:**
             a. Use Twitter Analytics to view engagement metrics across different times of the day and days of the week.
             b. Identify patterns where engagement peaks.
             c. Experiment with posting times based on these insights to verify if engagement increases.

         **3. Follower Demographics:**
         - **Approach:** Utilize Twitter's Audience Insights feature to gain a deeper understanding of your follower base.
         - **Actionable Steps:**
             a. Review demographic information such as location, gender, interests, and device usage.
             b. Tailor future content to match the interests and demographics of your audience.
             c. Consider creating persona profiles for key audience segments to better target future tweets.

         **4. Keyword Performance:**
         - **Approach:** Investigate how different keywords and hashtags perform in terms of visibility and engagement.
         - **Actionable Steps:**
             a. Use Twitter Analytics or a third-party tool to track the performance of specific keywords and hashtags.
             b. Compare engagement metrics for tweets with and without these keywords/hashtags.
             c. Identify which keywords/hashtags correlate with higher engagement andvisibility.
         **Recommendations for Future Strategy:**
         Based on the above analyses, you may find that certain themes/topics, posting times, or keywords resonate more with your audience. Use these insights to refine your Twitter strategy, focusing on content that drives engagement.
 Consider experimenting with different content formats (e.g., images, videos, polls) to further increase engagement. Regularly reviewing your analytics will help you stay aligned with your audience's preferences and improve your Twitter
 performance over time.</result>
 </task_result>

The result from the critic agent is:
 <task_evaluation>
    <goal>Analyze current Twitter performance and audience engagement.</goal>

    <outcome>PARTIALLY_MEETS_REQUIREMENT</outcome>

    <evaluation>
    The result provided does not fully meet the requirements of the task due to the agent's inability to directly access or analyze live Twitter analytics or any real-time data. However, the agent offers a structured approach and actionable
steps for conducting the analysis through available tools and best practices, which aligns with the task's goal to a certain extent.

    While the agent's response partially fulfills the task's need by providing a methodological framework for analysis, it does not directly deliver the analysis or insights that were expected from reviewing current Twitter analytics. The
task specifically aimed for an analysis of current performance and audience engagement based on historical data, which the agent did not provide.

    Yet, the guidance on how to perform content analysis, identify engagement patterns, understand follower demographics, and analyze keyword performance is valuable and relevant to the goal. The agent effectively translates the task
requirements into a DIY guide that the user can follow to achieve the desired analysis.

    In conclusion, while the agent's response does not directly offer the analysis and insights expected from the task, it provides a comprehensive guide for the user to perform the analysis independently. This approach partially meets the
task requirements by enabling the user to gather the necessary information, albeit indirectly.
    </evaluation>

    <tools_required>
        <tool>
            <name>Twitter Analytics</name>
            <description>Access to Twitter Analytics for detailed performance metrics.</description>
        </tool>
        <tool>
            <name>Third-Party Social Media Management Tools</name>
            <description>Use of tools like Hootsuite, Buffer, or Sprout Social for advanced analytics and scheduling.</description>
        </tool>
    </tools_required>
</task_evaluation>

---

The possible values inside the <outcome> tag are:
- MEETS_REQUIREMENT: the reslt meets the requirements of the task.
- PARTIALLY_MEETS_REQUIREMENT: the result partially meets the requirements of the task. Some aspects are missing or incomplete.
- DOES_NOT_MEET_REQUIREMENT: the result does not meet the requirements of the task. The result is inaccurate or incomplete.
- OTHER: for any other evaluation that does not fit the above categories. Provide a detailed explanation in the <evaluation> tag.

For the task and result below:
<task>
    <goal>{goal}</goal>
    <description>{description}</description>
</task>

<task_result>
    <goal>{goal}</goal>
    <result>{result}</result>
</task_result>

Please provide an evaluation of the result in the format described above. Ensure to output correctly formed XML tags (e.g. opening <task_evaluation> and closing </task_evaluation> tags).
"""

MARKDOWN_AGENT_SYSTEM_PROMPT = """You are an AI Agent with advanced markdown capabilities.
Your role is to assist users in summarising and formatting text which comes from the outputs of the run of multiple AI Agents trying to solve a given problem.
The input text will contain xml that represent the structure of the outputs from the agents. You are adept at transforming this xml into properly formatted markdown content.
"""

MARKDOWN_AGENT_PROMPT = """Transform the given xml content into markdown format. Follow the instructions below:
- Summarise the text in the <goal> and <objective> tags to come up with a concise title for the markdown content.
- Format the text in the <task_results> tags into appropriate markdown format. Each <task_result> in the <task_results> tag should be formatted as a separate section with a title, goal, and result.

---
For instance for the following input:
<problem>
It is my mother's 60th birthday in 2 weeks. I still haven't found a gift for her. Please help me find the right gift for her birthday.
</problem>

<objective>
Find the right gift for the user mother's 60th birthday.
</objective>

<tasks_results>
    <task_result>
        <goal>
        Gather information about the user mother's tastes, preferences, and needs to identify potential gift ideas.
        </goal>
        <result>
        I have compiled a detailed questionnaire with the following questions:
        1. What are your hobbies and favorite leisure activities?
        2. Is there something specific you have mentioned wanting or needing recently?
        3. What special memories or life milestones do you cherish?
        4. Do you prefer experiences over physical gifts?
        5. What are your preferences regarding jewelry, clothing sizes, favorite colors, authors, artists, or other personal tastes?
        </result>
    </task_result>

    <task_result>
        <goal>
        Generate a comprehensive document outlining gift ideas based on the answers from the questionnaire.
        </goal>
        <result>
    Here is the analysis of the responses to the Mother's 60th Birthday Gift Questionnaire and the identified gift ideas:

    **Summary of Questionnaire Responses**

    1. **Personal Interests:**
    - The mother enjoys gardening, reading, and exploring new hobbies like pottery.
    - She has been curious about learning more about photography and nature journaling.

    2. **Items She Needs:**
    - She mentioned needing a new set of gardening tools, as her current ones are quite worn.
    - She also commented on wanting a comfortable, stylish robe for relaxing at home.

    3. **Cherished Memories:**
    - The mother fondly recalls family vacations to the national parks when she was younger.
    - She is looking forward to celebrating her 60th birthday and the upcoming retirement of her husband.

    4. **Lifestyle:**
    - The mother prefers a mix of physical gifts and experiential gifts, as she values both material items and making memories.
    - She gravitates towards classic, timeless styles in clothing and jewelry.

    5. **Preferences:**
    - Her favorite colors are blues and greens, and she admires the work of impressionist painters.
    - She enjoys reading novels by authors such as Isabel Allende and Gabriel García Márquez.

    **Gift Categories and Ideas**

    1. **Hobbies and Interests:**
    - A pottery class or workshop for her to explore the new hobby
    - A high-quality camera and photography lesson package
    - A nature journaling kit with artisanal supplies

    2. **Household and Daily Needs:**
    - A premium gardening tool set with ergonomic handles
    - A luxurious, plush robe in her favorite colors

    3. **Cherished Memories and Experiences:**
    - A weekend getaway to a national park, reminiscent of her family vacations
    - A commissioned painting or artwork depicting a scenic landscape
    - A personalized photo album or scrapbook highlighting important milestones

    4. **Lifestyle and Preferences:**
    - A classic, timeless piece of jewelry, such as a necklace or bracelet
    - A collection of novels by her favorite authors, with a personalized note
    - A subscription to an art or literature-focused magazine
        </result>
    </task_result>
</tasks_results>

I expect the markdown content to be formatted as follows:
# Finding the right gift mother's 60th birthday

## Problem statement
It is my mother's 60th birthday in 2 weeks. I still haven't found a gift for her. Please help me find the right gift for her birthday.

## Objective
Find the right gift for my mother's 60th birthday.

# Tasks

## Gather information about mother's tastes, preferences, and needs

### Goal
Gather information about the user mother's tastes, preferences, and needs to identify potential gift ideas.

### Result
I have compiled a detailed questionnaire with the following questions:
1. What are your hobbies and favorite leisure activities?
2. Is there something specific you have mentioned wanting or needing recently?
3. What special memories or life milestones do you cherish?
4. Do you prefer experiences over physical gifts?
5. What are your preferences regarding jewelry, clothing sizes, favorite colors, authors, artists, or other personal tastes?

## Generate a comprehensive document outlining gift ideas

### Goal
Generate a comprehensive document outlining gift ideas based on the answers from the questionnaire.

### Result
Here is the analysis of the responses to the Mother's 60th Birthday Gift Questionnaire and the identified gift ideas:

Mother's 60th Birthday Gift Questionnaire Responses:
- **Personal Interests:** mother enjoys gardening, reading, and exploring new hobbies like pottery.
- **Items She Needs:** She mentioned needing a new set of gardening tools and a comfortable, stylish robe.
- **Cherished Memories:** fondly recalls family vacations to the national parks and upcoming retirement of her husband.
- **Lifestyle:** prefers a mix of physical gifts and experiential gifts, classic, timeless styles in clothing and jewelry.
- **Preferences:** favorite colors are blues and greens, admires impressionist painters, enjoys novels by Isabel Allende and Gabriel García Márquez.

Here are some ideas for gifts based on the questionnaire responses:
- **Hobbies and Interests:** pottery class, high-quality camera, nature journaling kit.
- **Household and Daily Needs:** premium gardening tool set, luxurious robe.
- **Cherished Memories and Experiences:** weekend getaway, commissioned painting, personalized photo album.
- **Lifestyle and Preferences:** classic, timeless jewelry, novels by favorite authors, art or literature-focused magazine.
---

Write the markdown content based on the input xml content provided below:
<problem>
{problem_statement}
</problem>

<objective>
{objective}
</objective>

{tasks_results}
"""
