These files contain the original data sent by the LIGO team with an additional fields on the end (fields 23:33) which has the first publication time as reported in Hydra in seconds after the published origin time.
The first publication time (field 23) was taken as the minimum of the following fields: 

TFIRSTEXTP: Time of SINSTFIR (initial external agency publication) (seconds after Origin time) 
TFIRSTATWCINT: Time of first ATWC PUBLIC publication. (seconds after Origin time)  
TFIRSTPTWCINT: Time of first PTWC PUBLIC publication. (seconds after Origin time)
TDETECTLATENCY: This is the detection latency for the hydra automatic system in seconds (i.e. tEventCreatedInHydra - tOrigin). This is the time that Hydra became aware of the event, regardless of the external source of the detection(GLASS, TWC, first-time-caller-long-time-listener, CIA)
TFIRSTPUB: The time (seconds after Origin time) that this event was first publicly released. (Publicly released means released to the world outside of Hydra, which under the current configuration may mean that it was only released to Chixilub.)

Fields 24:33
---- FINAL HYPOCENTER ----
24: TORIGINPDE	From the Verified PDE, this is the Origin time. (Decimal seconds since 1970-01-01)
25: MAGPDE	From the Verified PDE, this is the Preferred Magnitude.
26: DLATPDE	From the Verified PDE, this is the Latitude of the Hypocenter
27: DLONPDE	From the Verified PDE, this is the Longitude of the Hypocenter
28: DDEPTHPDE	From the Verified PDE, this is the Depth of the Hypocenter
---- INITIAL HYPOCENTER ----
29: TORIGININITIAL	From the initial NEIC public release of the Event, this is the Origin time.
30: MAGINITIAL 		From the initial NEIC public release of the Event, this is the Preferred Magnitude. (Decimal seconds since 1970-01-01)
31: DLATINITIAL		From the initial NEIC public release of the Event, this is the Latitude of the Hypocenter
32: DLONINITIAL		From the initial NEIC public release of the Event, this is the Longitude of the Hypocenter
33: DDEPTHINITIAL 	From the initial NEIC public release of the Event, this is the Depth of the Hypocenter

