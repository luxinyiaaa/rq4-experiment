# False-premise sensitivity

This analysis covers 89 false-premise questions. The primary abstention label separates silent from abstaining answers for both judges.

## Strict and lenient correctness

The lenient rule permits a Neutral premise-correction keypoint if every other decisive keypoint is entailed and no keypoint is contradicted. When the correction is itself the requested answer (AS-055-e61cb36aa7_2), lenient correctness equals strict correctness.

| model | judge | n | strict | previous_lenient | intended_lenient | answers_without_correction |
|---|---|---|---|---|---|---|
| egi | primary | 89 | 18/89 (20.2%) | 45/89 (50.6%) | 45/89 (50.6%) | 49/89 (55.1%) |
| openai | primary | 89 | 7/89 (7.9%) | 14/89 (15.7%) | 14/89 (15.7%) | 21/89 (23.6%) |
| openai_gpt56_sol | primary | 89 | 61/89 (68.5%) | 81/89 (91.0%) | 81/89 (91.0%) | 23/89 (25.8%) |
| openai_gpt6_astra | primary | 89 | 61/89 (68.5%) | 75/89 (84.3%) | 75/89 (84.3%) | 16/89 (18.0%) |
| egi | secondary | 89 | 16/89 (18.0%) | 45/89 (50.6%) | 45/89 (50.6%) | 52/89 (58.4%) |
| openai | secondary | 89 | 5/89 (5.6%) | 16/89 (18.0%) | 16/89 (18.0%) | 26/89 (29.2%) |
| openai_gpt56_sol | secondary | 89 | 59/89 (66.3%) | 79/89 (88.8%) | 79/89 (88.8%) | 23/89 (25.8%) |
| openai_gpt6_astra | secondary | 89 | 60/89 (67.4%) | 75/89 (84.3%) | 75/89 (84.3%) | 17/89 (19.1%) |

## Four premise outcomes

| model | judge | n | corrects the premise | accepts the premise | silent on the premise | abstains |
|---|---|---|---|---|---|---|
| egi | primary | 89 | 21/89 (23.6%) | 29/89 (32.6%) | 31/89 (34.8%) | 8/89 (9.0%) |
| openai | primary | 89 | 8/89 (9.0%) | 35/89 (39.3%) | 23/89 (25.8%) | 23/89 (25.8%) |
| openai_gpt56_sol | primary | 89 | 62/89 (69.7%) | 3/89 (3.4%) | 24/89 (27.0%) | 0/89 (0.0%) |
| openai_gpt6_astra | primary | 89 | 63/89 (70.8%) | 1/89 (1.1%) | 18/89 (20.2%) | 7/89 (7.9%) |
| egi | secondary | 89 | 16/89 (18.0%) | 28/89 (31.5%) | 35/89 (39.3%) | 10/89 (11.2%) |
| openai | secondary | 89 | 6/89 (6.7%) | 33/89 (37.1%) | 23/89 (25.8%) | 27/89 (30.3%) |
| openai_gpt56_sol | secondary | 89 | 61/89 (68.5%) | 3/89 (3.4%) | 25/89 (28.1%) | 0/89 (0.0%) |
| openai_gpt6_astra | secondary | 89 | 63/89 (70.8%) | 2/89 (2.2%) | 18/89 (20.2%) | 6/89 (6.7%) |

## Correcting keypoints

| question_id | correcting_keypoint | false_premise |
|---|---|---|
| AS-002-ac70e580f4_2 | The OCG team also includes the OCG Executive Committee | The team consists only of network leaders; the page also includes the OCG Executive Committee. |
| AS-005-98a245cac8_2 | Researchers can search for and use already produced products instead of always requesting a new run | Claims a new model run is always required; the page says researchers can also search for and use already produced products. |
| AS-006-7e426acee5_2 | GOOS is led by UNESCO’s Intergovernmental Oceanographic Commission, not the World Meteorological Organization. | The question says the World Meteorological Organization leads GOOS; the passage identifies it as a co-sponsor and names UNESCO’s Intergovernmental Oceanographic Commission as the leader. |
| AS-012-9869eaeb23_2 | ReLTER, not CrocoTile, enables searching and accessing DEIMS-SDR data from within R code | Attributes ReLTER's ability to search DEIMS-SDR from R code to CrocoTile. |
| AS-017-ed593aa4fb_1 | CEA and the Cyprus Institute led the drone-based methane pilot, not the University of Helsinki. | Attributes the drone-based methane pilot to the University of Helsinki; the page assigns it to CEA and the Cyprus Institute. |
| AS-022-5a33d02c3f_1 | The Canadian slowdown instruction is not the voluntary programme; NOAA's US Slow Zones programme is voluntary. | Describes Canada's vessel-slowdown instruction as voluntary, whereas the page identifies the US NOAA Slow Zones programme as voluntary. |
| AS-024-eaaee24105_2 | Private-sector organisations are not limited to being EPOS platform users | Private-sector organisations can only use the EPOS platform; the page also describes supplier and customer roles. |
| AS-026-f1f8a763e9_1 | The nominal analysis time is 12:00Z for the previous day, not 13:30Z on the current day. | Confuses the image datasets’ daily update time of approximately 13:30Z with the ensemble map’s nominal analysis time of 12:00Z on the previous day. |
| AS-028-c874413ce7_2 | The final list will come into force only with publication of the new GCOS Implementation Plan in 2028, not in 2025 | Claims the list came into force in 2025; the passage says 2025 was the public-review period and that the final list will take effect only with the 2028 Implementation Plan. |
| AS-029-e78f65d4c9_1 | The Technical Committee, not the Communications Committee, oversees standards and protocols. | Assigns oversight of technical standards and protocols to the Communications Committee; the page assigns it to the Technical Committee. |
| AS-032-caf471a49f_2 | ENVRI is the Environmental Sciences cluster, not the Life Sciences cluster | Describes ENVRI as the Life Sciences cluster; the page identifies ENVRI as the Environmental Sciences cluster. |
| AS-033-6cdf7122cc_1 | The sensor-data tool concerns Flemish LifeWatch, not the LifeWatchGreece Micro-CT lab | Attributes the Flemish LifeWatch sensor-data tool to the LifeWatchGreece Micro-CT virtual lab. |
| AS-035-5bc7ddc9a5_2 | The team aims to increase participation by scientists and experts from non-represented countries, not already represented countries. | Says the team seeks greater participation from countries already represented in OCG networks; the stated goal concerns non-represented countries. |
| AS-036-3146fb14c4_1 | AGORA-Lab was established in 2020, not 2005 | Places AGORA-Lab's establishment in 2005; the page says 2005 was when LOA and CIMEL extended their cooperation to aerosol LiDAR technologies, while AGORA-Lab was established in 2020. |
| AS-040-ef494b6237_1 | Exploratory Platforms are labelled as a whole, not separately for each component | Claims Exploratory Platforms are labelled separately for each component; the page assigns component-by-component labelling to observational platforms. |
| AS-042-b5cbf8a14a_2 | ICOS metadata is marked CC0 1.0, not CC BY 4.0 | Claims that ICOS metadata is released under CC BY 4.0; the page assigns CC BY 4.0 to ICOS data and marks metadata CC0 1.0. |
| AS-049-90ec4d09d9_2 | ICOS National Networks coordinate the stations, not universities alone | Claims that universities alone coordinate ICOS stations; the passage says ICOS National Networks coordinate them. |
| AS-050-a827a43098_1 | The ENVRI-Hub demonstrator, not Blue-Cloud, was described as ENVRI-FAIR’s major achievement. | Calls Blue-Cloud ENVRI-FAIR’s main achievement; the programme identifies the ENVRI-Hub demonstrator instead. |
| AS-053-97312c0f38_1 | Institutions operate the research sites and platforms, not the Topic Centres and Head Office | Attributes operation of the physical network to the Topic Centres and Head Office; the page says institutions operate its sites and platforms. |
| AS-055-e61cb36aa7_2 | The 2003 IOC Assembly session was held in Paris, not Maputo. | Places the 2003 IOC Assembly session in Maputo; the passage places that session in Paris and describes an earlier GOOS workshop in Maputo. |
| AS-057-c17bb95544_1 | ISGC 2025 is in Taipei, not Heidelberg | Places ISGC 2025 in Heidelberg, although the page explicitly gives Taipei as its location. |
| AS-059-882159aecd_2 | DEIMS-SDR is not limited to eLTER sites | Claims that the catalogue covers only eLTER sites; the page says its coverage is not exclusive to them. |
| AS-060-d69e8d2992_1 | The country forms a National Consortium; the General Assembly does not form it. | The ACTRIS General Assembly forms a joining country's National Consortium; the text says the country begins by forming it. |
| AS-064-2d7fc4bc78_1 | The International Ocean Carbon Coordination Project (IOCCP), not GLODAPv2, acts as the GOOS Biogeochemistry Panel | Identifies GLODAPv2, a named data synthesis product, as the GOOS Biogeochemistry Panel. |
| AS-066-1254ca870f_2 | The third API is for raw ecosystem data, not processed atmospheric data. | Claims the third API is for processed atmospheric data; the page explicitly says it is for raw ecosystem data. |
| AS-070-46f4d6e8f7_2 | Participating ships provide observations voluntarily at no cost, not for payment. | Claims participating ships are paid to submit observations; the page says they provide them voluntarily at no cost. |
| AS-071-122b0e73e4_1 | RItrainPlus is coordinated by Università degli Studi di Milano-Bicocca, not European universities collectively | Says European universities collectively coordinate RItrainPlus; the page names Università degli Studi di Milano-Bicocca as coordinator. |
| AS-073-622317da41_2 | Teams began outlining Exemplar projects during breakout working sessions, not the plenary | Places the initial outlining of Exemplar projects in the plenary; the passage places it in breakout working sessions. |
| AS-076-af6f4a5778_2 | The Interim Council has delegates from 19 member countries, not 21 | Attributes the 21 countries politically supporting eLTER to the Interim Council’s membership; the Council has delegates from 19 member countries. |
| AS-077-b28f7f252b_1 | A National Distributed Centre locally organises each workshop, not the Service Centre | Says the Service Centre locally organises each workshop; the page assigns that role to a National Distributed Centre. |
| AS-081-456ce4ca3e_1 | Many stations were not fully reporting as needed by international centres. | Claims most stations were already reporting enough; the workshop said many were not fully reporting as needed by international centres. |
| AS-083-b0c2949b26_2 | The conference begins Tuesday 10 September, not Monday 9 September | Says the conference starts on Monday 9 September; the conference starts on Tuesday 10 September, while Monday is for a networking event. |
| AS-085-656d196dbd_1 | GSRN is a new network being implemented; GRUAN was established over 20 years ago. | Attributes GRUAN's establishment over 20 years ago to the new GSRN. |
| AS-086-93c30376f8_2 | Fixed, ground-based stations are Observational Platforms, not Exploratory Platforms | Describes Exploratory Platforms as fixed, ground-based stations; the page identifies those stations as Observational Platforms. |
| AS-088-fe4ecff3e7_1 | The lead institution was the University of Helsinki, not ICOS. | Names ICOS as the lead institution, although the page names the University of Helsinki; ICOS is mentioned in connection with site co-location. |
| AS-095-f1c1fd1835_2 | Data from 12 stations, not 17, were reprocessed for the historic InGOS N₂O series. | Attributes the 17 reprocessed stations of the adjacent historic InGOS CH₄ series to the N₂O series, which had 12. |
| AS-096-28d67e2287_1 | Observing Together, not CoastPredict, connects community and national initiatives to the global system | Attributes Observing Together’s work connecting community and national initiatives to CoastPredict. |
| AS-099-9ddbef938d_2 | The first demonstration case concerns non-indigenous and invasive species, not native species | Says the first demonstration case concerns native species; the page explicitly identifies non-indigenous and invasive species. |
| AS-105-1c27251498_2 | IDC ITALIA SRL, not the Digital Europe program, coordinated GREAT | Says the Digital Europe program coordinated GREAT, whereas the page identifies it as the funding program and names IDC ITALIA SRL as coordinator. |
| AS-106-27db1dc417_1 | Data and services are provided within the TCS and made interoperable with the central ICS-C hub, not provided by the hub within the TCS. | Reverses the stated relationship between data and service provision within the TCS and the central ICS-C hub. |
| AS-110-e8dbc8158d_2 | The Service Centre is in Lecce, Italy, not Amsterdam | Places the Service Centre in Amsterdam; the page places it in Lecce, Italy, and places the Virtual Lab and Innovation Centre in Amsterdam. |
| AS-111-9be4ddcc44_1 | The ocean absorbs roughly a quarter of excess human CO2 emissions each year, not 93% | Assigns the 93% excess-heat figure to excess human CO2 emissions; the passage gives their annual uptake as roughly a quarter. |
| AS-114-1efe9ff8c3_2 | The Ocean Best Practices System is a system for disseminating and archiving best practices, not a GOOS observing network. | Describes the Ocean Best Practices System as a GOOS observing network; the page describes it as a system for disseminating and archiving best practices. |
| AS-117-8eea671fd9_2 | The Science Committee, not the Data Management Team, elects the Executive Committee | Claims that the Data Management Team elects the Executive Committee; the passage says the Science Committee does. |
| AS-119-d62cfd4c8b_1 | During the negative SOI phase, pressure is below normal at Tahiti and above normal at Darwin, not the reverse | Reverses the explicitly stated pressure conditions at Tahiti and Darwin during the negative SOI phase. |
| AS-122-77c1985954_2 | Reliability, expertise, partnerships and engaged professionals are drivers of success, not the core values | Treats the stated drivers of success as the strategy’s core values. |
| AS-127-1f1189073d_1 | Geological collection data are published through GeoCASe, not IPT. | Assigns IPT to geological collection data; the page assigns GeoCASe to those data. |
| AS-132-7756dde488_2 | The project ends on September 30, 2026, not September 30, 2022 | States that Geo-INQUIRE ends on September 30, 2022; the page states that it ends on September 30, 2026. |
| AS-137-4ce608a062_2 | 21 countries politically support eLTER RI's establishment, not 27 | Treats the 27 countries involved in developing eLTER RI as the number providing political support; the page states that 21 countries provide political support. |
| AS-140-253f40623d_2 | GRASP is for the South-East Pacific, not the South-West Pacific. | Describes GRASP as the alliance for the South-West Pacific; the page places it in the South-East Pacific. |
| AS-141-7d33f826d7_1 | Stratospheric ozone has a cooling effect, not a warming effect. | Says stratospheric ozone warms the atmosphere; the page explicitly says it has a cooling effect. |
| AS-143-70523109d6_2 | The task supported transitioning HF radar systems to a sustained effort, not keeping them short-term | Claims the task aimed to keep HF radar systems short-term; the text says it supported their transition to a sustained effort. |
| AS-144-db70f3182a_1 | OCEATLAN is associated with buoy arrays, not database integration | Attributes NEARGOOS's database-integration focus to OCEATLAN, which the text associates with buoy arrays. |
| AS-146-a978b4303f_2 | GOOS is led by the Intergovernmental Oceanographic Commission, not directly by UNESCO | Describes GOOS as UNESCO-led; the passage says it is IOC-led and that IOC is part of UNESCO. |
| AS-147-5454df9ab6_1 | The group was established by people actively engaged in eLTER work, not by the eLTER Head Office | The eLTER Head Office established the group; the text says people actively engaged in eLTER work established it, while the Head Office supports it. |
| AS-152-89338ec71c_2 | The central hub belongs to the Integrated Core Services (ICS-C), not the Thematic Core Services. | Places the central hub in the Thematic Core Services; the page places it in the Integrated Core Services (ICS-C). |
| AS-153-43cb7187f0_1 | Basel was an additional network city, not a pilot city | Basel was selected as a small pilot city; it was an additional network city, while Zürich was the small pilot. |
| AS-155-4ecd7576e8_2 | The access point covers different data provider domains, not only one | Claims the central access point covers only one provider domain; the page says it covers different data provider domains. |
| AS-170-55cee6473c_1 | Category 1 sites apply prime methods in at least two spheres, not necessarily every sphere | Claims that Category 1 sites must use prime methods across every ecosystem sphere; the stated requirement is prime methods in at least two spheres. |
| AS-179-7cda613b16_1 | ICOS Data Portal has three levels of data and data products, not two | Claims that ICOS Data Portal has two levels of data and data products; the page says it has three. |
| AS-183-77f84702c4_1 | Ocean is an ICOS domain alongside Atmosphere and Ecosystem | Claims that ICOS does not work in the Ocean domain, although the page explicitly names Ocean as one of its three domains. |
| AS-185-b719bd5ea5_2 | The lake network has more than 600 water-level reference stations, not more than 300. | Attributes more than 300 water-level reference stations to the lake network; the lake row specifies more than 600, while the river row mentions more than 300 river-discharge reference stations. |
| AS-187-36beacc13a_1 | OceanOPS tracks over 100,000 observations a day, not fewer | Claims OceanOPS tracks fewer than 100,000 observations a day; the page says it tracks over 100,000. |
| AS-189-fca8e3c6a0_2 | Collaborative Jupyter notebooks and notebooks for anonymous use are separate services | Conflates the collaborative Jupyter notebooks with the separate notebooks offered for anonymous use. |
| AS-190-9dbafac48b_1 | The announced course runs from 26–29 June 2023, not 11–16 September 2023 | Treats the September dates for the repeat as the dates of the announced June course. |
| AS-192-a5f2a15467_2 | The event also allows online participation | Claims the event is open only to in-person attendees; the page explicitly says online participation is also available. |
| AS-195-fb1cbec228_1 | Virtual access does not require a selection process and is available to unlimited users. | Claims that ACTRIS virtual access requires competitive selection because resources are limited; the page says virtual access needs no selection, while physical access has a competitive selection process. |
| AS-198-87cc0b8396_2 | CLU uses Cloudnet, not EARLINET | Claims that CLU uses EARLINET; the page says CLU uses Cloudnet and associates EARLINET with ARES. |
| AS-199-faa2a39dfe_1 | WP8 covers the atmosphere subdomain, not the marine subdomain | Assigns the marine subdomain to WP8; the page assigns atmosphere to WP8 and marine to WP9. |
| DR-002-45b7adbfaa_2 | The Scottish Environment Protection Agency, Stirling Office, holds the data, not the British Oceanographic Data Centre | Names the British Oceanographic Data Centre as the data holder; the record identifies it as the collating centre and names the Scottish Environment Protection Agency’s Stirling Office as the data holder. |
| DR-003-edb664c96e_1 | The automatic meteorological data acquisition system was installed in 1983, not 1984. | The question says the system was installed in 1984; the record says it was installed in 1983. |
| DR-005-105d9654d2_2 | The Flemish Environment Agency holds the data, while the Belgian Marine Data Centre is the collating centre. | The Belgian Marine Data Centre holds the observations; the record identifies it as the collating centre and names the Flemish Environment Agency as the data holding centre. |
| DR-006-b501dc1334_1 | The surveys covered 7 sites in 1988, not 15. | Assigns the 1992 count of 15 sites to 1988, when the record gives 7 sites for 1988. |
| DR-009-c711eb68a5_1 | The measurement height is 14.0 m, not 12.0 m. | Treats the site's 12.0 m altitude as the measurement height; the series is identified as being measured at 14.0 m. |
| DR-010-7deb5bb27a_2 | The data holding centre is the Scottish Environment Protection Agency, Stirling Office, not the British Oceanographic Data Centre | Identifies the British Oceanographic Data Centre as the data holding centre; the record names it as the collating centre instead. |
| DR-011-f45f8910a2_1 | The University of Liège, Laboratory of Oceanology holds the data, not the Belgian Marine Data Centre | Says the Belgian Marine Data Centre holds the data; the record names it as the collating centre and names the University of Liège, Laboratory of Oceanology as the data holding centre. |
| DR-012-8098458a11_1 | The recorded study period was 1970–1972, not 1970–1973 | States that the recorded period ended in 1973; the record says it ended in 1972. |
| DR-013-e26bf0de6e_2 | The 1991 cruise sampled 21 stations, not 27 | Assigns the 1992 count of 27 stations to the 1991 cruise, which sampled 21. |
| DR-014-0632f4f70e_1 | The record is not ongoing. | Describes the monitoring as ongoing, whereas the record explicitly says it is not. |
| DR-017-dd1abc2e14_2 | The originator was IHE, Management Unit of North Sea and Scheldt Estuary Mathematical Models, not the Belgian Marine Data Centre | Identifies the Belgian Marine Data Centre, listed as the collating centre, as the study's originator. |
| DR-018-f694bc2c67_1 | No data were gathered in 1999 or 2000 | Claims uninterrupted annual measurements through 2000, although the record states that no data were gathered in 1999 or 2000. |
| DR-020-38b06799a7_2 | The observations are returned to the Marine Conservation Society for input and analysis, not to the British Oceanographic Data Centre for analysis | Assigns analysis to the British Oceanographic Data Centre, which the record names as the collating centre; the observations are returned to the Marine Conservation Society for input and analysis. |
| DR-022-ae04c418f0_1 | The European model’s grid spacing is 25–30 km, not about 150 km | Assigns the Global Model’s approximately 150 km grid spacing to the European Wave Model. |
| DR-024-4171d2fb26_2 | Queens University Belfast’s Department of Agriculture and Environmental Science holds the data, not the British Oceanographic Data Centre | Identifies the British Oceanographic Data Centre as the data holder; the record identifies it as the collating centre. |
| DR-025-a366646373_1 | The data-holding centre is FPS Economy, Belgian Continental Shelf, not the Belgian Marine Data Centre. | Identifies the Belgian Marine Data Centre as the data holder; the record names it as the collating centre and identifies FPS Economy, Belgian Continental Shelf as the data holder. |
| DR-028-fe12b6e899_1 | The data concern a single industrial discharge, not multiple discharges | Says the surveys cover multiple industrial discharges; the record explicitly limits the data to one. |
| DR-034-0b056b10e4_2 | The archive covers 2021–2024, not 2021–2025 | Claims that the Renon archive covers 2021–2025; its stated coverage is 2021–2024. |
| DR-045-bc6e0c28f4_2 | Gillan Harbour is on the Lizard Peninsula in Cornwall, not on the Isles of Scilly | Places Gillan Harbour on the Isles of Scilly rather than on the Lizard Peninsula in Cornwall. |
| DR-047-45b032a472_2 | The collection is not ongoing. | States that the suspended-matter data are part of an ongoing collection; the record says they are not ongoing. |
