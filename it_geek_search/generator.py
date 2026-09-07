from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
import pandas as pd

PARTICIPANTS = ["Rohan", "Aisha", "Vivek", "Meera", "Karan", "Nisha", "Arjun", "Priya"]

# ---------------------------------------------------------------------------
# COMBINATORIAL DIALOGUE POOLS
# To ensure massive diversity across 4200 messages without repeating exact
# conversations, we build sessions dynamically: 1 Starter + N Middles + 1 Ending.
# ---------------------------------------------------------------------------

TOPIC_COMPONENTS = {
    "hackathon": {
        "starters": [
            "Guys, are we participating in the upcoming hackathon this weekend?",
            "Hackathon registrations open ho gayi hain, kaunsa problem statement le rahe hain?",
            "Kya sab log hackathon submission ke liye ready hain?",
            "Inter-college hackathon aa raha hai, team banayein?",
            "Next month ke AI hackathon ka idea sochna padega.",
            "Registration deadline aaj raat 12 baje tak hai hackathon ki.",
        ],
        "middles": [
            "Haan bilkul, I was thinking we can build an AI search tool.",
            "Mast idea hai! Tech stack kya hoga? Python aur FastAPI chalega?",
            "Backend main dekh lunga, database ke liye vector store setup karna padega.",
            "UI ka design Figma pe ready kar deti hoon aaj shaam tak.",
            "Main pitch deck aur presentation slides par kaam shuru karti hoon.",
            "Sustainability track interesting lag raha hai — carbon footprint calculator banana hai.",
            "Flutter se mobile app bhi bana sakte hain, cross-platform rahega.",
            "Database ke liye Firebase ya Supabase use karenge? Quick setup chahiye.",
            "CTO of a fintech startup aur Google Developer Expert judges hain.",
            "WebSocket server ready hai, latency 50ms se neeche rakhni padegi.",
            "Frontend pe voice input bhi add karna hai, Web Speech API use karenge.",
            "Backend API tests sab pass ho gaye, 95% code coverage hai.",
            "Frontend responsive hai, mobile aur desktop dono pe test kiya.",
            "DevPost pe project submit karna hai midnight tak, README likhi hai?",
            "GitHub repo public karna mat bhoolna submission ke liye.",
            "Code freeze 2 hours pehle hoga submission se.",
            "Testing framework Jest ya Mocha use karenge.",
        ],
        "endings": [
            "Code-athon final karlo sab log, venue college auditorium rahega.",
            "Done deal! Let's meet early Saturday morning with our laptops and chargers.",
            "Great! Project demo ekdam crisp aur functional rakhna hai.",
            "Cool, let's vote on the idea by tonight.",
            "Presentation 7 minutes ki hogi, slides max 10 rakhni hain.",
            "Demo video record karke backup rakhte hain, live demo fail hone ka risk hai.",
            "Agle hackathon mein first aana hai, preparation start karte hain.",
            "WiFi backup plan rakhna hai venue pe.",
        ]
    },
    "server": {
        "starters": [
            "AWS ka monthly bill aa gaya hai, unexpected spike dikh raha hai.",
            "Production server ka CPU 95% pe chal raha hai, auto-scaling trigger nahi ho raha.",
            "GCP se AWS migrate karne ka discussion karte hain, cost comparison ready hai.",
            "Kubernetes cluster mein pod restarts bahut zyaada ho rahe hain.",
            "CDN cache hit ratio bahut kam hai, 40% se neeche gir gaya.",
            "Elasticsearch nodes memory limit hit kar rahe hain.",
        ],
        "middles": [
            "Kaunsa service high cost generate kar raha hai? EC2 ya RDS?",
            "OpenSearch cluster aur NAT gateway usage kaafi badh gaya hai.",
            "Unused staging environments band kar dete hain, instant 40% saving hogi.",
            "Let's also downsize t3.xlarge instances to t3.medium for dev.",
            "CloudWatch alarm bhi set kar diya hai agar billing threshold cross ho.",
            "ASG group ki max capacity 4 thi, 8 tak badhao abhi.",
            "Load balancer health checks fail ho rahe the, target group unhealthy dikh raha tha.",
            "Config update karke reload karo, downtime nahi aayega.",
            "GCP pe monthly 1800 dollars lag rahe hain, AWS pe estimate 1200 aaya hai.",
            "Migration ke liye Terraform scripts likhne padenge, 2 weeks lagenge.",
            "Memory limit low set thi pods pe, OOMKilled status dikh raha hai.",
            "Helm chart update karo values.yaml mein, memory 512Mi se 1Gi karo.",
            "Horizontal pod autoscaler bhi configure karo CPU threshold 70% pe.",
            "Cache-Control headers galat set hain, max-age=0 lag raha hai kuch routes pe.",
            "Static assets ke liye 1 year cache set karo, fingerprinted filenames hain.",
            "Log rotation configure karo disk space ke liye.",
            "VPN access revoke karo ex-employees ka.",
        ],
        "endings": [
            "Server cost aur infra ka budget check karna padega, monthly cap 200 dollars set karte hain.",
            "Monitoring dashboard pe alert rules bhi tighten kar dete hain.",
            "Client ko bhi inform karna padega maintenance window ke baare mein.",
            "Prometheus metrics check karo, Grafana dashboard pe spikes dikhengi.",
            "CloudFront invalidation run karo deployment ke baad.",
            "Uptime SLA 99.9% maintain karna hai.",
            "Security patches pending hain 3 servers pe, update plan karo.",
        ]
    },
    "deployment": {
        "starters": [
            "Production deployment scheduled hai tonight at 11 PM.",
            "CI/CD pipeline fail ho gayi, Docker image build nahi ho raha.",
            "Blue-green deployment setup karna hai zero downtime ke liye.",
            "Hotfix deploy karna hai urgently, payment gateway timeout issue hai.",
            "Feature flag system integrate karna hai, LaunchDarkly ya homegrown?",
            "Next release ka tag create karna hai GitHub pe.",
        ],
        "middles": [
            "QA testing complete ho chuki hai, all critical regression tests passed.",
            "Database migration script verify kar li hai staging environment mein?",
            "Haan rollback script bhi ready rakhi hai just in case koi failure aaye.",
            "Main logs monitor karunga deployment ke waqt Datadog par.",
            "Base image deprecated ho gayi thi, node:16 se node:20 pe upgrade karo.",
            "Dockerfile optimize karo, multi-stage build se image size 200MB aa jayega.",
            "ALB target groups create kar diye hain blue aur green ke liye.",
            "Health check endpoint ready hai, /api/health pe 200 return karega.",
            "Traffic switch script ready hai, 10% canary se start karenge.",
            "Fix ready hai PR #287 mein, code review fast track karo.",
            "Cherry-pick karke release branch pe merge kar do.",
            "Unleash self-hosted setup kar deta hoon Docker compose ke saath.",
            "New payment module ko feature flag ke peeche rakhte hain.",
            "API versioning check karo backward compatibility ke liye.",
            "Performance benchmark run karo before/after.",
        ],
        "endings": [
            "Smooth release ho gayi, zero downtime! Good job everyone.",
            "Users ko release notes update send kar do changelog ke saath.",
            "Pipeline green ho gayi, staging pe deploy ho raha hai.",
            "Rollback procedure document kar do runbook mein.",
            "Incident timeline document karo post-mortem ke liye.",
            "Gradual rollout 5% users se start karenge, monitoring ke saath.",
            "Stakeholder email bhejo release summary ke saath.",
        ]
    },
    "trip": {
        "starters": [
            "Guys bahut stress ho gaya hai, let's plan a weekend getaway!",
            "Goa plan bana rahe hain Christmas ke around, kisi ko interest hai?",
            "Trekking plan hai iss weekend, Triund jaane ka mann hai.",
            "Office outing approve ho gayi hai, Rishikesh ja rahe hain next month!",
            "Road trip plan karte hain Jaipur tak, heritage sites dekhenge.",
            "Long weekend aa raha hai, kahan ghoomne chalna hai?",
        ],
        "middles": [
            "Yes please! Kahan chalna hai? Mountains ya beach?",
            "Chalo Manali fix hai, booking start karo before rates go up.",
            "Travel budget manageable hai, per head 4000 aayega including fuel.",
            "Weather forecast check kiya? It might snow, carry warm jackets!",
            "Maine hotel shortlist kar liya hai near Mall Road, booking link share karti hoon.",
            "South Goa peaceful rahega, Palolem beach ke paas villa milega.",
            "Flight tickets book kar lo jaldi, prices badh rahe hain December ke.",
            "Triund moderate trek hai, 6 hours lagenge round trip.",
            "Camping gear hire karna padega Dharamkot se, tents aur sleeping bags.",
            "River rafting confirmed hai, Grade 3 rapids pe karenge.",
            "Bungee jumping bhi add karo itinerary mein, Jumpin Heights famous hai.",
            "Amber Fort aur Hawa Mahal must visit hain.",
            "Local food toh try karna hi padega — dal baati churma aur pyaaz kachori.",
            "Google Maps pe route offline save kar lo.",
            "Cash bhi rakhna, sab jagah UPI nahi chalega.",
        ],
        "endings": [
            "Saturday early morning nikalte hain drive karke, traffic avoid hoga.",
            "Awesome, Friday evening sab log pack karke ready rehna.",
            "Main restaurant reservations dekh leta hoon, seafood try karna hai.",
            "Morning 5 baje start karna padega, sunset tak wapas aa jayenge.",
            "First aid kit le jana important hai adventure activities ke liye.",
            "Night stay ke liye haveli-style hotel book karte hain, aesthetic photos aayengi.",
            "Group photo lena mat bhoolna har jagah.",
        ]
    },
    "banter": {
        "starters": [
            "Coffee break anyone? Pantry me fresh snacks aaye hain.",
            "Kisi ne naya office chair try kiya? Ergonomic wali jo aayi hai?",
            "IPL ka match dekha kal? Last ball pe six mara!",
            "Aaj Monday blues hit kar raha hai, motivation kahan se laayein?",
            "Kisi ke paas phone charger hai Type-C wala? Battery 2% hai.",
            "Aaj ka lunch kahan se order karna hai? Biryani?",
        ],
        "middles": [
            "Kal raat Champions League match kisne dekha? What a game!",
            "Mera code build fail ho gaya tha, pure time wahi debug kar rahi thi.",
            "Chai pe chalo, 10 min break is needed after that brutal meeting.",
            "Swiggy pe 50% discount chal raha hai, let's order together.",
            "Lumbar support ekdum mast hai, back pain kaafi kam ho gaya.",
            "Standing desk bhi mangwao, alternate karna chahiye sitting aur standing.",
            "Spotify playlist share karo koi, coding ke waqt music chahiye.",
            "Crazy match tha yaar, aise finishes rare hote hain.",
            "Office fantasy league mein meri team top pe hai, prize pool kitna hai?",
            "Friday karib aa raha hai bhai, bas 4 din aur survive karo.",
            "Tab tak chai tapri pe chalo, wahan ki cutting chai best hai.",
            "Mere desk pe hai charger, aa ja le ja. MacBook wala bhi hai.",
            "Amazon pe sale chal rahi hai electronics pe, 40% off hai.",
            "Netflix pe koi acchi series recommend karo.",
        ],
        "endings": [
            "Weekend pe movie dekhne ka plan banayein?",
            "Lo-fi beats playlist bhej raha hoon, focus ke liye best hai.",
            "Cricket tournament office mein bhi organize karte hain monsoon mein.",
            "Board game night plan karte hain Wednesday ko, Catan ya Uno?",
            "Thanks yaar, lifesaver ho tum. Ek chai toh banti hai.",
            "Secret Santa plan karte hain December mein.",
            "Team photo update karo website pe.",
        ]
    },
    "code_review": {
        "starters": [
            "PR #142 open hai for authentication refactor, please review.",
            "Notification service ka PR review kar do, #198 pe hai.",
            "Payment gateway integration ka code review karna hai, PR #225.",
            "Search API ka PR #250 ready hai, Elasticsearch queries optimize ki hain.",
            "GraphQL schema redesign ka PR review chahiye, #272.",
            "Frontend state management refactoring, PR #310 is up.",
        ],
        "middles": [
            "Looking into it now. JWT token expiry time thoda extend kar sakte hain?",
            "Password hashing mein bcrypt rounds increase karne chahiye for better security.",
            "Firebase Cloud Messaging integration clean hai, retry logic add karo.",
            "Rate limiting bhi add karo, per user max 100 notifications per hour.",
            "Razorpay webhook signature verification missing hai, add karo.",
            "Idempotency key implement karo duplicate payments avoid karne ke liye.",
            "Fuzzy matching ka fuzziness parameter configurable banana chahiye.",
            "Scroll API use kar raha hoon deep pagination ke liye.",
            "N+1 query problem dikhi hai user.posts resolver mein, DataLoader use karo.",
            "Depth limiting add karo queries pe, DOS attack prevent hoga.",
            "Type safety improve karo, any hatao.",
            "API response time 200ms se under rakhna hai.",
            "Bundle size check karo webpack analyzer se.",
        ],
        "endings": [
            "Done, suggestions address kar diye hain aur tests update kar diye.",
            "LGTM! Merging to main branch now. Good refactoring work!",
            "Looks good, approved! Production deploy kar do.",
            "Unit tests payment scenarios cover kar rahe hain? Edge cases bhi.",
            "Test coverage 85% se 92% pe aa gayi hai, edge cases add kiye.",
            "DataLoader implementation kar diya, batch queries 80% reduce ho gayi.",
            "Documentation update karo API changes ke saath.",
        ]
    },
    "database": {
        "starters": [
            "Postgres slow query logs check kiye? Index missing lag raha hai users table par.",
            "MongoDB collection size 50GB cross kar gayi hai, sharding enable karna padega.",
            "Database migration production pe run karna hai, Flyway ya Liquibase?",
            "Time-series data store karne ke liye TimescaleDB evaluate kar rahe hain.",
            "DynamoDB read/write capacity units ka cost bahut zyaada aa raha hai.",
            "Redis cache instances memory limit cross kar rahe hain.",
        ],
        "middles": [
            "Compound index add karne se query latency 200ms se 15ms drop ho gayi.",
            "Redis cache TTL kitna rakha hai session tokens ke liye?",
            "Data backup automated snapshot daily 2 AM pe verify ho raha hai na?",
            "Shard key kya rakhein? tenant_id se even distribution hogi.",
            "Migration chunk size 64MB default se 128MB karo, faster balancing hogi.",
            "Flyway simple hai, SQL-based migrations aur version tracking dono milte hain.",
            "Rollback scripts likhna mat bhoolna har migration ke saath.",
            "Hypertable create karo timestamp column pe, automatic chunking hogi.",
            "Retention policy set karo 90 days ka, purana data compress hoga.",
            "On-demand capacity mode se provisioned mode pe switch karo predictable load ke liye.",
            "DAX caching layer add karo read-heavy workloads ke liye.",
            "Connection string rotate karo quarterly.",
            "Read replicas add karo reporting queries ke liye.",
            "Query plan explain analyze karo slow queries pe.",
        ],
        "endings": [
            "24 hours rakha hai, with automatic eviction policy configured.",
            "Monitoring ke liye MongoDB Atlas alerts configure kar do.",
            "Migration successful rahi, zero data loss aur 3 seconds downtime.",
            "Grafana dashboard connect kar do real-time visualization ke liye.",
            "Single-table design adopt karo, access patterns pehle define karo.",
            "VACUUM analyze run karo weekly basis pe.",
            "Data masking implement karo PII columns pe.",
        ]
    },
    "interview": {
        "starters": [
            "Backend engineer candidate ka technical interview scheduled hai 3 PM.",
            "Frontend developer ki vacancy ke liye 50+ applications aayi hain.",
            "DevOps engineer ka final round hai aaj, panel mein kaun kaun hai?",
            "Data scientist position ke liye interview loop plan karo.",
            "Internship program ke liye college recruitment drive plan kar rahe hain.",
            "Mobile developer candidate ki take-home assignment submit ho gayi hai.",
        ],
        "middles": [
            "System design round mein caching aur rate limiting focus karenge.",
            "Coding challenge mein data structures aur concurrency handling check kar lenge.",
            "Resume screening mein 12 shortlist kiye, React aur TypeScript experience dekha.",
            "Take-home assignment prepare kiya hai — simple dashboard build karna hai.",
            "Main aur Karan hain panel mein, Kubernetes aur CI/CD focus rahega.",
            "Live exercise mein Docker multi-stage build troubleshoot karwayenge.",
            "Case study mein recommendation engine design karwayenge.",
            "Statistical knowledge test bhi include karo, A/B testing concepts cover karo.",
            "Top 5 engineering colleges mein campus visit schedule karo.",
            "Online coding test HackerRank pe set up kar do, 500 students expected hain.",
            "Feedback 24 hours mein submit karna mandatory hai.",
            "Panel bias training complete karo.",
            "Diversity metrics track karo hiring pipeline mein.",
        ],
        "endings": [
            "Feedback form submit kar diya hai, candidate strong hai Python mein.",
            "Offer letter draft karke HR ko forward kar dete hain.",
            "Culture fit round bhi add karo, team dynamics important hai.",
            "Salary negotiation ke liye HR se baat kar lo, budget range share karo.",
            "Interview rubric finalize karke sab interviewers ko share karo.",
            "PPT aur company intro presentation update karo latest projects ke saath.",
            "Onboarding checklist ready karo new hires ke liye.",
        ]
    },
    "office_event": {
        "starters": [
            "Annual hackathon showcase next Friday town hall mein hoga.",
            "Diwali celebration office mein plan kar rahe hain, ideas do.",
            "Quarter end celebration hai Friday ko, team lunch plan karo.",
            "Company anniversary celebration mein talent show organize kar rahe hain.",
            "Team building workshop hai next Wednesday, escape room jaana hai.",
            "New Year party ka venue decide karna hai.",
        ],
        "middles": [
            "Demo slides aur recorded video ready rakhna hai sab teams ko.",
            "Prizes announce ho gaye hain, top 3 teams ko vouchers milenge!",
            "Rangoli competition rakh sakte hain, team-wise participation hogi.",
            "Sweet distribution aur decoration ka budget 15000 hai.",
            "Theme ethnic wear rakha hai, sab traditional mein aayenge.",
            "Italian restaurant book kiya hai Cyber Hub mein, 1 PM slot.",
            "Award ceremony bhi hogi, best performer ka announcement hoga.",
            "Registrations open hain — singing, dancing, standup comedy sab include hai.",
            "Judges panel mein CEO, HR head aur ek external artist honge.",
            "Mystery Rooms ka slot book kiya hai, 2 batches mein jayenge.",
            "Lunch ke baad bowling bhi plan hai, alley paas mein hai.",
            "Parking arrangements check karo venue pe.",
            "Dietary restrictions ka form circulate karo.",
        ],
        "endings": [
            "Hamari team ka presentation 4th slot pe hai, rehearse kar lete hain.",
            "Photography booth setup karo, props ke saath fun photos aayengi.",
            "Transport arrange karo office se restaurant tak, cab pool banana padega.",
            "Rehearsal space book karo conference room 3 mein evenings ke liye.",
            "Attendance confirm karo by Monday, headcount finalize karna hai.",
            "Social media pe post karo event highlights.",
            "Leftover food NGO ko donate karo.",
        ]
    },
    "budget": {
        "starters": [
            "Q2 tooling budget approve ho gaya leadership se.",
            "Annual cloud infrastructure budget review time hai.",
            "New hire ke liye laptop procurement budget chahiye.",
            "Training budget allocate karna hai team ke liye is quarter.",
            "SaaS subscriptions audit karo, unused tools band karo.",
            "Marketing tech stack ka budget final approve ho gaya.",
        ],
        "middles": [
            "Github Copilot enterprise licenses renew karwaye ya GitHub standard?",
            "Enterprise better hai with audit logging and seat management.",
            "Last year 2.4 lakh dollars spend hua, is year estimate 3 lakh hai.",
            "Reserved instances buy karo 1-year commitment pe, 30% saving milegi.",
            "MacBook Pro M3 ka current price 1.8 lakh hai corporate discount ke saath.",
            "Peripherals bhi include karo — monitor, keyboard, mouse, headset.",
            "Udemy Business per seat 400 dollars annual hai, Coursera 500 dollars.",
            "Conference travel budget bhi allocate karo, 2 conferences per engineer.",
            "Figma, Notion, Slack, Jira, Confluence — sab active use mein hain.",
            "Miro aur Loom ke 8 seats hain but 3 hi use karte hain, downgrade karo.",
            "Invoice approval TAT 5 business days hai.",
            "Tax implications check karo international purchases pe.",
            "Budget variance report share karo leadership ke saath.",
        ],
        "endings": [
            "Done, invoice finance team ko bhej di hai payment release ke liye.",
            "CFO se meeting schedule karo next week, proposal present karna hai.",
            "IT asset management system mein entry karo purchase ke baad.",
            "Total training budget 8000 dollars approved, distribute karo evenly.",
            "Procurement team ko updated vendor list bhej do renegotiation ke liye.",
            "Forecast accuracy improve karo historical data se.",
            "Audit trail maintain karo sab transactions ka.",
        ]
    },
}

CHAT_SLANG_FILLERS = [
    "bhai", "yaar", "sahi hai", "pakka", "chalega", "dekhte hain",
    "done", "cool", "tension mat lo", "ekdum", "haaha", "arre", "waah",
    "bilkul", "chal", "haan", "accha", "thik hai", "chalo", "suno",
]

TYPO_MAP = {
    "please": ["pls", "plz"],
    "meeting": ["meting", "meetin"],
    "tomorrow": ["tomrw", "tmrw"],
    "thanks": ["thx", "tysm"],
    "because": ["coz", "cuz"],
    "karna": ["krna", "krrna"],
    "hai": ["h", "haii"],
    "theek": ["thk", "thek"],
    "kya": ["kyaaa", "kya"],
    "nahi": ["nhi", "nai"],
    "raha": ["rha", "raha"],
    "hain": ["hain", "h"],
    "bahut": ["bhot", "bahot"],
    "accha": ["acha", "achaa"],
}

def inject_noise(text: str, rng: random.Random) -> str:
    words = text.split()
    # Randomly add filler or slang
    if rng.random() < 0.25:
        filler = rng.choice(CHAT_SLANG_FILLERS)
        if rng.random() < 0.5:
            words.insert(0, filler + ",")
        else:
            words.append(filler)

    # Randomly introduce common chat abbreviations/typos
    for i, w in enumerate(words):
        w_lower = w.lower().strip(".,?!")
        if w_lower in TYPO_MAP and rng.random() < 0.35:
            replacement = rng.choice(TYPO_MAP[w_lower])
            words[i] = replacement

    return " ".join(words)


def generate_chat_dataset(seed: int = 42, n_messages: int = 4200) -> pd.DataFrame:
    """
    Generates a realistic synthetic corpus spanning 6 months.
    Uses combinatorial generation to ensure unique conversation flows.
    """
    rng = random.Random(seed)
    start_date = datetime(2025, 1, 1, 9, 0, 0)
    end_date = datetime(2025, 6, 30, 21, 0, 0)
    total_duration_sec = int((end_date - start_date).total_seconds())

    topics = list(TOPIC_COMPONENTS.keys())
    data_rows: List[Dict[str, Any]] = []

    # Anchored special scenarios for verified benchmark evaluation
    special_sessions: List[Tuple[datetime, str, List[Tuple[str, str]]]] = [
        (
            datetime(2025, 1, 20, 14, 0, 0),
            "hackathon",
            [
                ("Rohan", "Guys, what is the plan for the upcoming hackathon?"),
                ("Aisha", "We should build a multilingual search tool for local Indian languages."),
                ("Vivek", "Sounds awesome. Let's lock in the stack with Python and ChromaDB."),
                ("Arjun", "code-athon final karlo sab log, auditorium book ho chuka hai weekend ke liye!"),
                ("Priya", "Great, let's assemble at 9 AM on Saturday."),
            ],
        ),
        (
            datetime(2025, 2, 14, 18, 30, 0),
            "trip",
            [
                ("Meera", "Are we actually going on the road trip or is it getting postponed again?"),
                ("Priya", "Manali trip ka plan final ho gaya, stay aur cab ka budget finalize kar liya."),
                ("Rohan", "chalo Manali fix hai, booking start karo guys!"),
                ("Karan", "We confirmed 3 days in Old Manali, hotel is right next to Mall Road."),
                ("Aisha", "Awesome, booking advance paid. Departure on Friday midnight."),
            ],
        ),
        (
            datetime(2025, 3, 12, 11, 15, 0),
            "server",
            [
                ("Rohan", "Checking AWS billing dashboard today, our spend crossed the threshold."),
                ("Karan", "The elasticsearch instances and high memory databases are draining our cloud budget."),
                ("Priya", "Rohan mentioned cutting costs by terminating idle compute nodes immediately."),
                ("Vivek", "I am switching us to spot instances on Kubernetes to reduce compute expenses by 60%."),
                ("Rohan", "Approved, let's keep infrastructure expenses strictly under our monthly limit."),
            ],
        ),
        (
            datetime(2025, 3, 28, 13, 19, 30),
            "hackathon",
            [
                ("Aisha", "Hackathon prep meeting — final tech stack freeze karte hain."),
                ("Vivek", "Python + FastAPI backend aur React frontend, ChromaDB for vectors."),
                ("Nisha", "Main pitch deck finalize kar rahi hoon, USP highlight karna padega."),
                ("Priya", "Prizes announce ho gaye hain, top 3 teams ko vouchers milenge!"),
                ("Vivek", "Hamari team ka presentation 4th slot pe h, rehearse kar lete hain."),
            ],
        ),
        (
            datetime(2025, 4, 10, 15, 0, 0),
            "database",
            [
                ("Arjun", "Database latency was spiking above 400ms during peak query traffic."),
                ("Karan", "Added composite index on tenant_id and created_at columns."),
                ("Aisha", "Performance improved by 85%, response times are sub-20ms now."),
                ("Vivek", "Redis connection pooling was also bottlenecked, bumped pool size to 50."),
            ],
        ),
        (
            datetime(2025, 5, 18, 22, 10, 0),
            "deployment",
            [
                ("Karan", "Alert triggered! Service auth-worker reporting 502 bad gateway errors."),
                ("Arjun", "Memory leak in connection handling caused container OOM kill."),
                ("Rohan", "Hotfix deployed to production at 22:45, memory usage stabilized."),
                ("Aisha", "Post-mortem meeting scheduled tomorrow at 10 AM."),
            ],
        ),
        (
            datetime(2025, 6, 5, 16, 0, 0),
            "budget",
            [
                ("Priya", "Q3 software license procurement needs sign-off before Friday."),
                ("Rohan", "I reviewed the JetBrains and GitHub enterprise renewal quotes, within expectations."),
                ("Meera", "Cloud credit grant got approved too, saving us 5000 USD."),
                ("Karan", "Finance approved the hardware upgrades for the local test bench."),
            ],
        ),
    ]

    msg_id = 0
    session_counter = 0

    # Insert special sessions first
    for dt, topic, dialog in special_sessions:
        session_id = f"session_{session_counter:04d}"
        session_counter += 1
        current_time = dt
        for sender, text in dialog:
            data_rows.append(
                {
                    "message_id": msg_id,
                    "session_id": session_id,
                    "sender": sender,
                    "timestamp": pd.Timestamp(current_time),
                    "text": text,
                    "topic": topic,
                }
            )
            msg_id += 1
            current_time += timedelta(minutes=rng.randint(1, 7), seconds=rng.randint(0, 59))

    # Generate remaining messages up to n_messages across realistic combinatorial sessions
    while msg_id < n_messages:
        session_id = f"session_{session_counter:04d}"
        session_counter += 1

        random_offset = rng.randint(0, total_duration_sec)
        session_start = start_date + timedelta(seconds=random_offset)
        topic = rng.choice(topics)

        # Select participants for this session
        session_participants = rng.sample(PARTICIPANTS, min(8, max(4, rng.randint(3, 7))))

        # Build dynamic dialogue: 1 starter + k middles + 1 ending
        pool = TOPIC_COMPONENTS[topic]
        num_middles = rng.randint(2, min(5, len(pool["middles"])))
        
        dialogue_lines = []
        dialogue_lines.append(rng.choice(pool["starters"]))
        dialogue_lines.extend(rng.sample(pool["middles"], num_middles))
        dialogue_lines.append(rng.choice(pool["endings"]))

        curr_time = session_start
        last_sender = None

        for base_text in dialogue_lines:
            if msg_id >= n_messages:
                break
                
            # Pick a sender (try not to let the same person talk twice in a row)
            possible_senders = [p for p in session_participants if p != last_sender]
            if not possible_senders:
                possible_senders = session_participants
            sender = rng.choice(possible_senders)
            last_sender = sender
            
            noisy_text = inject_noise(base_text, rng)

            data_rows.append(
                {
                    "message_id": msg_id,
                    "session_id": session_id,
                    "sender": sender,
                    "timestamp": pd.Timestamp(curr_time),
                    "text": noisy_text,
                    "topic": topic,
                }
            )
            msg_id += 1
            curr_time += timedelta(minutes=rng.randint(1, 8), seconds=rng.randint(5, 55))

    df = pd.DataFrame(data_rows).sort_values("timestamp").reset_index(drop=True)
    df["message_id"] = range(1, len(df) + 1)
    return df


def generate_benchmark_queries() -> List[Dict[str, Any]]:
    # Contains the exact same 40 benchmark queries as before.
    return [
        { "id": "Q01", "prompt": "Where did the group settle on having the coding marathon meetup?", "type": "semantic", "zero_overlap": True, "target_topic": "hackathon", "target_content": "auditorium book ho chuka hai weekend ke liye", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q02", "prompt": "mountain vacation destination decision", "type": "semantic", "zero_overlap": True, "target_topic": "trip", "target_content": "chalo Manali fix hai", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q03", "prompt": "strategies to minimize cloud hosting expenditure", "type": "semantic", "zero_overlap": True, "target_topic": "server", "target_content": "terminating idle compute nodes immediately", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q04", "prompt": "optimizing slow SQL query execution times", "type": "semantic", "zero_overlap": True, "target_topic": "database", "target_content": "Added composite index on tenant_id", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q05", "prompt": "critical crash due to out of memory leak in production", "type": "semantic", "zero_overlap": False, "target_topic": "deployment", "target_content": "Memory leak in connection handling caused container OOM kill", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q06", "prompt": "midday meal food delivery discount", "type": "semantic", "zero_overlap": True, "target_topic": "banter", "target_content": "Swiggy pe 50% discount chal raha hai", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q07", "prompt": "strengthening login authentication credential privacy", "type": "semantic", "zero_overlap": True, "target_topic": "code_review", "target_content": "Password hashing mein bcrypt rounds increase", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q08", "prompt": "evaluating software engineering applicants skills", "type": "semantic", "zero_overlap": True, "target_topic": "interview", "target_content": "Backend engineer candidate ka technical interview", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q09", "prompt": "financial funding grant for virtual machine costs", "type": "semantic", "zero_overlap": True, "target_topic": "budget", "target_content": "Cloud credit grant got approved too, saving us 5000 USD", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q10", "prompt": "annual company competition demonstration time slot", "type": "semantic", "zero_overlap": True, "target_topic": "office_event", "target_content": "Hamari team ka presentation 4th slot pe hai", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q11", "prompt": "What did Rohan say about server costs?", "type": "attributed", "zero_overlap": False, "target_topic": "server", "target_content": "Server cost aur infra ka budget check karna padega", "expected_sender": "Rohan", "start_date": None, "end_date": None },
        { "id": "Q12", "prompt": "What did Priya announce regarding the Manali trip?", "type": "attributed", "zero_overlap": False, "target_topic": "trip", "target_content": "Manali trip ka plan final ho gaya", "expected_sender": "Priya", "start_date": None, "end_date": None },
        { "id": "Q13", "prompt": "What suggestion did Vivek make regarding Kubernetes and spot instances?", "type": "attributed", "zero_overlap": False, "target_topic": "server", "target_content": "switching us to spot instances on Kubernetes", "expected_sender": "Vivek", "start_date": None, "end_date": None },
        { "id": "Q14", "prompt": "What feedback did Meera give about the candidate?", "type": "attributed", "zero_overlap": False, "target_topic": "interview", "target_content": "Feedback form submit kar diya hai, candidate strong hai", "expected_sender": "Meera", "start_date": None, "end_date": None },
        { "id": "Q15", "prompt": "What did Arjun report about database latency improvement?", "type": "attributed", "zero_overlap": False, "target_topic": "database", "target_content": "Compound index add karne se query latency", "expected_sender": "Arjun", "start_date": None, "end_date": None },
        { "id": "Q16", "prompt": "What did Karan notice about the AWS bill?", "type": "attributed", "zero_overlap": False, "target_topic": "server", "target_content": "AWS ka monthly bill aa gaya hai, unexpected spike", "expected_sender": "Karan", "start_date": None, "end_date": None },
        { "id": "Q17", "prompt": "What did Aisha mention about the weather in Manali?", "type": "attributed", "zero_overlap": False, "target_topic": "trip", "target_content": "Weather forecast check kiya? It might snow", "expected_sender": "Aisha", "start_date": None, "end_date": None },
        { "id": "Q18", "prompt": "What did Nisha say about prizes for the hackathon?", "type": "attributed", "zero_overlap": False, "target_topic": "office_event", "target_content": "Annual hackathon showcase next Friday", "expected_sender": "Nisha", "start_date": None, "end_date": None },
        { "id": "Q19", "prompt": "What did Rohan say about the deployment status?", "type": "attributed", "zero_overlap": False, "target_topic": "deployment", "target_content": "Smooth release ho gayi, zero downtime", "expected_sender": "Rohan", "start_date": None, "end_date": None },
        { "id": "Q20", "prompt": "What did Priya mention about license procurement?", "type": "attributed", "zero_overlap": False, "target_topic": "budget", "target_content": "Q3 software license procurement needs sign-off", "expected_sender": "Priya", "start_date": None, "end_date": None },
        { "id": "Q21", "prompt": "What discussions took place in March regarding server costs?", "type": "temporal", "zero_overlap": False, "target_topic": "server", "target_content": "Checking AWS billing dashboard today", "expected_sender": None, "start_date": "2025-03-01", "end_date": "2025-03-31" },
        { "id": "Q22", "prompt": "What trip planning happened in February?", "type": "temporal", "zero_overlap": False, "target_topic": "trip", "target_content": "Manali trip ka plan final ho gaya", "expected_sender": None, "start_date": "2025-02-01", "end_date": "2025-02-28" },
        { "id": "Q23", "prompt": "What hackathon decisions were made in January?", "type": "temporal", "zero_overlap": False, "target_topic": "hackathon", "target_content": "code-athon final karlo sab log", "expected_sender": None, "start_date": "2025-01-01", "end_date": "2025-01-31" },
        { "id": "Q24", "prompt": "What database indexing optimizations were completed in April?", "type": "temporal", "zero_overlap": False, "target_topic": "database", "target_content": "Added composite index on tenant_id", "expected_sender": None, "start_date": "2025-04-01", "end_date": "2025-04-30" },
        { "id": "Q25", "prompt": "What production outage or alert occurred in May?", "type": "temporal", "zero_overlap": False, "target_topic": "deployment", "target_content": "Alert triggered! Service auth-worker reporting 502", "expected_sender": None, "start_date": "2025-05-01", "end_date": "2025-05-31" },
        { "id": "Q26", "prompt": "What budget and license approvals happened in June?", "type": "temporal", "zero_overlap": False, "target_topic": "budget", "target_content": "Q3 software license procurement needs sign-off", "expected_sender": None, "start_date": "2025-06-01", "end_date": "2025-06-30" },
        { "id": "Q27", "prompt": "What travel plans were discussed before March 2025?", "type": "temporal", "zero_overlap": False, "target_topic": "trip", "target_content": "Manali trip ka plan final ho gaya", "expected_sender": None, "start_date": "2025-01-01", "end_date": "2025-02-28" },
        { "id": "Q28", "prompt": "Any code review discussions in January 2025?", "type": "temporal", "zero_overlap": False, "target_topic": "code_review", "target_content": "PR #142 open hai", "expected_sender": None, "start_date": "2025-01-01", "end_date": "2025-01-31" },
        { "id": "Q29", "prompt": "What infrastructure monitoring was set up in March?", "type": "temporal", "zero_overlap": False, "target_topic": "server", "target_content": "CloudWatch alarm bhi set kar diya hai", "expected_sender": None, "start_date": "2025-03-01", "end_date": "2025-03-31" },
        { "id": "Q30", "prompt": "What happened on February 14th regarding our holiday?", "type": "temporal", "zero_overlap": False, "target_topic": "trip", "target_content": "chalo Manali fix hai, booking start karo", "expected_sender": None, "start_date": "2025-02-14", "end_date": "2025-02-15" },
        { "id": "Q31", "prompt": "Rohan AWS EC2 and OpenSearch billing costs", "type": "hybrid", "zero_overlap": False, "target_topic": "server", "target_content": "OpenSearch cluster aur NAT gateway usage", "expected_sender": "Karan", "start_date": None, "end_date": None },
        { "id": "Q32", "prompt": "Priya Hotel booking Mall road Manali package", "type": "hybrid", "zero_overlap": False, "target_topic": "trip", "target_content": "hotel shortlist kar liya hai near Mall Road", "expected_sender": "Priya", "start_date": None, "end_date": None },
        { "id": "Q33", "prompt": "JWT token expiry and bcrypt password hashing security", "type": "hybrid", "zero_overlap": False, "target_topic": "code_review", "target_content": "JWT token expiry time thoda extend kar sakte hain", "expected_sender": None, "start_date": None, "end_date": None },
        { "id": "Q34", "prompt": "Postgres database latency composite index tenant_id", "type": "hybrid", "zero_overlap": False, "target_topic": "database", "target_content": "Compound index add karne se query latency 200ms se 15ms", "expected_sender": "Arjun", "start_date": None, "end_date": None },
        { "id": "Q35", "prompt": "502 bad gateway auth-worker memory leak container OOM", "type": "hybrid", "zero_overlap": False, "target_topic": "deployment", "target_content": "Service auth-worker reporting 502 bad gateway errors", "expected_sender": "Karan", "start_date": None, "end_date": None },
        { "id": "Q36", "prompt": "GitHub Copilot enterprise licenses renewal quote", "type": "hybrid", "zero_overlap": False, "target_topic": "budget", "target_content": "Github Copilot enterprise licenses renew karwaye", "expected_sender": "Priya", "start_date": None, "end_date": None },
        { "id": "Q37", "prompt": "College auditorium venue Saturday 9am code-athon", "type": "hybrid", "zero_overlap": False, "target_topic": "hackathon", "target_content": "code-athon final karlo sab log, auditorium book ho chuka hai", "expected_sender": "Arjun", "start_date": None, "end_date": None },
        { "id": "Q38", "prompt": "Redis cache TTL session tokens 24 hours eviction", "type": "hybrid", "zero_overlap": False, "target_topic": "database", "target_content": "Redis cache TTL kitna rakha hai session tokens ke liye", "expected_sender": "Priya", "start_date": None, "end_date": None },
        { "id": "Q39", "prompt": "Swiggy lunch order food discount coupon code", "type": "hybrid", "zero_overlap": False, "target_topic": "banter", "target_content": "Swiggy pe 50% discount chal raha hai", "expected_sender": "Meera", "start_date": None, "end_date": None },
        { "id": "Q40", "prompt": "Spot instances Kubernetes cluster cost reduction 60 percent", "type": "hybrid", "zero_overlap": False, "target_topic": "server", "target_content": "switching us to spot instances on Kubernetes", "expected_sender": "Vivek", "start_date": None, "end_date": None },
    ]
