"""Response envelope and data models for DB-backed endpoints.

Per ADR-010: camelCase field names everywhere (identical in Python and JSON).
Per ADR-020: datetime fields are UTC ISO-8601 with Z suffix.

Pydantic v2 models with ConfigDict(extra="forbid") on all request/param models.
Response models use extra="ignore" so the serialisation layer doesn't reject
extra DB columns (they route to `extras`).

ruff: noqa: N815  (canonical fields use weewx camelCase: outTemp, windSpeed, etc.)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def utc_isoformat(dt: datetime) -> str:
    """Serialise a UTC datetime to ISO-8601 with Z suffix (ADR-020)."""
    # Pydantic serialises datetime to "+00:00" by default; we want "Z".
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Observation / ArchiveRecord
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """Canonical observation (ADR-010 §3.1 + OpenAPI Observation schema).

    Full stock weewx column set per the user directive 2026-05-06: every column
    in STOCK_COLUMN_MAP is first-class here.  Operator-custom columns route
    through `extras`; stock weewx columns NEVER appear in `extras`.

    All numeric fields Optional — weather data is genuinely missing sometimes.
    `extras` is always present (may be empty).

    ruff: noqa: N815  (weewx camelCase names per ADR-010)
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    timestamp: str  # UTC ISO-8601 with Z

    # Core wview observation fields
    outTemp: float | None = None
    outHumidity: float | None = None
    windSpeed: float | None = None
    windDir: float | None = None
    windGust: float | None = None
    windGustDir: float | None = None
    barometer: float | None = None
    pressure: float | None = None
    altimeter: float | None = None
    dewpoint: float | None = None
    windchill: float | None = None
    heatindex: float | None = None
    rainRate: float | None = None
    rain: float | None = None
    radiation: float | None = None
    UV: float | None = None
    inTemp: float | None = None
    inHumidity: float | None = None

    # wview_extended core fields
    ET: float | None = None
    hail: float | None = None
    hailRate: float | None = None
    appTemp: float | None = None
    cloudbase: float | None = None
    cloudcover: float | None = None
    windrun: float | None = None
    maxSolarRad: float | None = None
    sunshineDur: float | None = None
    daySunshineDur: float | None = None
    rainDur: float | None = None
    THSW: float | None = None
    humidex: float | None = None
    pop: float | None = None
    illuminance: float | None = None
    noise: float | None = None

    # Lightning fields (wview_extended)
    lightning_strike_count: float | None = None
    lightning_distance: float | None = None
    lightning_noise_count: float | None = None
    lightning_disturber_count: float | None = None

    # Snow fields (wview_extended)
    snow: float | None = None
    snowDepth: float | None = None
    snowRate: float | None = None

    # Wind summary fields
    vecdir: float | None = None
    gustdir: float | None = None
    vecavg: float | None = None
    rms: float | None = None

    # Degree-days
    heatdeg: float | None = None
    cooldeg: float | None = None

    # Sensor expansion slots (wview_extended)
    extraTemp1: float | None = None
    extraTemp2: float | None = None
    extraTemp3: float | None = None
    extraHumid1: float | None = None
    extraHumid2: float | None = None
    soilTemp1: float | None = None
    soilTemp2: float | None = None
    soilTemp3: float | None = None
    soilTemp4: float | None = None
    soilMoist1: float | None = None
    soilMoist2: float | None = None
    soilMoist3: float | None = None
    soilMoist4: float | None = None
    leafTemp1: float | None = None
    leafTemp2: float | None = None
    leafWet1: float | None = None
    leafWet2: float | None = None

    # Electrical / system telemetry
    consBatteryVoltage: float | None = None
    heatingVoltage: float | None = None
    referenceVoltage: float | None = None
    supplyVoltage: float | None = None
    rxCheckPercent: float | None = None

    # Operator-custom columns (stock weewx columns NEVER appear here)
    extras: dict[str, Any] = {}
    source: str = "weewx"


class ArchiveRecord(Observation):
    """ArchiveRecord = Observation + interval (ADR-010 §3.2)."""

    interval: int


# ---------------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------------


class PageInfo(BaseModel):
    """Pagination metadata matching OpenAPI PageInfo schema."""

    cursor: str | None = None
    next: str | None = None
    previous: str | None = None
    limit: int
    page: int | None = None
    totalPages: int | None = None
    totalRecords: int | None = None


class ObservationResponse(BaseModel):
    """ObservationResponse envelope."""

    data: Observation | None
    units: dict[str, str]
    source: str
    generatedAt: str  # UTC ISO-8601 with Z


class ArchiveResponse(BaseModel):
    """ArchiveResponse envelope."""

    data: list[ArchiveRecord]
    units: dict[str, str]
    source: str
    generatedAt: str
    page: PageInfo


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class RecordEntry(BaseModel):
    """One named record (e.g. "All-time high outTemp")."""

    label: str
    canonicalField: str
    value: float | None = None
    observedAt: str | None = None  # UTC ISO-8601 with Z
    brokenInLast30Days: bool = False


class RecordsBundle(BaseModel):
    """Records grouped by section."""

    period: str
    sections: dict[str, list[RecordEntry]]


class RecordsResponse(BaseModel):
    """RecordsResponse envelope."""

    data: RecordsBundle
    units: dict[str, str]
    source: str
    generatedAt: str


# ---------------------------------------------------------------------------
# Reports (NOAA)
# ---------------------------------------------------------------------------


class ReportEntry(BaseModel):
    """One NOAA report file entry (monthly or yearly)."""

    kind: str  # "monthly" | "yearly"
    year: int
    month: int | None = None
    filename: str
    modifiedAt: str  # UTC ISO-8601 with Z


class ReportIndex(BaseModel):
    """Index of available NOAA reports."""

    reports: list[ReportEntry]


class ReportIndexResponse(BaseModel):
    """ReportIndexResponse envelope."""

    data: ReportIndex
    generatedAt: str


class NOAAReport(BaseModel):
    """Raw monthly NOAA report text."""

    year: int
    month: int
    filename: str
    rawText: str
    modifiedAt: str  # UTC ISO-8601 with Z


class NOAAYearlyReport(BaseModel):
    """Raw yearly NOAA report text."""

    year: int
    filename: str
    rawText: str
    modifiedAt: str  # UTC ISO-8601 with Z


class ReportResponse(BaseModel):
    """ReportResponse envelope."""

    data: NOAAReport
    generatedAt: str


class YearlyReportResponse(BaseModel):
    """YearlyReportResponse envelope."""

    data: NOAAYearlyReport
    generatedAt: str


# ---------------------------------------------------------------------------
# Almanac
# ---------------------------------------------------------------------------


class SunSnapshot(BaseModel):
    """Sun data block within AlmanacSnapshot."""

    rise: str | None = None
    set: str | None = None
    transit: str | None = None
    civilTwilightDawn: str | None = None
    civilTwilightDusk: str | None = None
    azimuth: float | None = None
    altitude: float | None = None
    rightAscension: float | None = None
    declination: float | None = None
    daylightMinutes: int | None = None
    daylightDeltaVsYesterdayMinutes: int | None = None
    nextEquinox: str | None = None
    nextSolstice: str | None = None


class MoonSnapshot(BaseModel):
    """Moon data block within AlmanacSnapshot."""

    rise: str | None = None
    set: str | None = None
    transit: str | None = None
    azimuth: float | None = None
    altitude: float | None = None
    rightAscension: float | None = None
    declination: float | None = None
    phaseName: str | None = None
    illuminationPercent: float | None = None
    nextFullMoon: str | None = None
    nextNewMoon: str | None = None


class AlmanacSnapshot(BaseModel):
    """AlmanacSnapshot matching OpenAPI AlmanacSnapshot schema."""

    date: str
    sun: SunSnapshot
    moon: MoonSnapshot


class AlmanacResponse(BaseModel):
    """AlmanacResponse envelope."""

    data: AlmanacSnapshot
    generatedAt: str


class SunTimesDay(BaseModel):
    """One day in a SunTimesSeries."""

    date: str
    sunrise: str | None = None
    sunset: str | None = None
    daylightMinutes: int | None = None


class SunTimesSeries(BaseModel):
    """SunTimesSeries matching OpenAPI schema."""

    year: int
    days: list[SunTimesDay]


class SunTimesResponse(BaseModel):
    """SunTimesResponse envelope."""

    data: SunTimesSeries
    generatedAt: str


class MoonPhaseDay(BaseModel):
    """One day in a MoonPhaseCalendar."""

    date: str
    phaseName: str
    illuminationPercent: float


class MoonPhaseCalendar(BaseModel):
    """MoonPhaseCalendar matching OpenAPI schema."""

    year: int
    month: int | None = None
    days: list[MoonPhaseDay]


class MoonPhaseResponse(BaseModel):
    """MoonPhaseResponse envelope."""

    data: MoonPhaseCalendar
    generatedAt: str


# ---------------------------------------------------------------------------
# Station
# ---------------------------------------------------------------------------


class StationMetadata(BaseModel):
    """StationMetadata matching OpenAPI StationMetadata schema."""

    stationId: str
    name: str
    latitude: float
    longitude: float
    altitude: float
    timezone: str
    timezoneOffsetMinutes: int
    unitSystem: str
    firstRecord: str | None = None
    lastRecord: str | None = None
    hardware: str | None = None


class StationResponse(BaseModel):
    """StationResponse envelope."""

    data: StationMetadata
    units: dict[str, str]
    generatedAt: str


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class WeewxColumnEntry(BaseModel):
    """One weewx archive column entry in CapabilityRegistry."""

    canonicalField: str
    archiveColumn: str


class CapabilityDeclaration(BaseModel):
    """Per ADR-038: one configured provider module."""

    providerId: str
    domain: str
    suppliedCanonicalFields: list[str]
    geographicCoverage: str
    defaultPollIntervalSeconds: int | None = None
    operatorNotes: str | None = None
    tileUrlTemplate: str | None = None
    wmsEndpointUrl: str | None = None
    wmsLayerName: str | None = None
    tileContentType: str | None = None
    iframeUrl: str | None = None


class CapabilityRegistry(BaseModel):
    """CapabilityRegistry matching OpenAPI CapabilityRegistry schema."""

    providers: list[CapabilityDeclaration]
    weewxColumns: list[WeewxColumnEntry]
    canonicalFieldsAvailable: list[str]


class CapabilityResponse(BaseModel):
    """CapabilityResponse envelope."""

    data: CapabilityRegistry
    generatedAt: str


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


class PageMetadata(BaseModel):
    """PageMetadata matching OpenAPI PageMetadata schema."""

    slug: str
    name: str
    icon: str
    navPosition: int
    builtIn: bool
    hidden: bool = False


class PageList(BaseModel):
    """PageList matching OpenAPI PageList schema."""

    pages: list[PageMetadata]


class PageListResponse(BaseModel):
    """PageListResponse envelope."""

    data: PageList
    generatedAt: str


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


class ChartGroup(BaseModel):
    """ChartGroup matching OpenAPI ChartGroup schema."""

    id: str
    name: str
    builtIn: bool
    members: list[str]
    defaultRange: str | None = None


class ChartGroupList(BaseModel):
    """ChartGroupList matching OpenAPI ChartGroupList schema."""

    groups: list[ChartGroup]


class ChartGroupResponse(BaseModel):
    """ChartGroupResponse envelope."""

    data: ChartGroupList
    generatedAt: str


# ---------------------------------------------------------------------------
# Content (markdown passthrough)
# ---------------------------------------------------------------------------


class MarkdownContent(BaseModel):
    """MarkdownContent matching OpenAPI MarkdownContent schema."""

    markdown: str
    updatedAt: str | None = None


class MarkdownResponse(BaseModel):
    """MarkdownResponse envelope."""

    data: MarkdownContent
    generatedAt: str


# ---------------------------------------------------------------------------
# Forecast (ADR-007, canonical-data-model §3.3, §3.4, §3.5, §3.10)
# ---------------------------------------------------------------------------

# ruff: noqa: N815  (field names use canonical camelCase: validTime, outTemp, etc.)


class HourlyForecastPoint(BaseModel):
    """Canonical hourly forecast record (ADR-010 §3.3, OpenAPI HourlyForecastPoint schema).

    extra="ignore" so provider-specific wire fields don't break normalization.
    Required fields per OpenAPI: validTime, source.
    All forecast numeric fields are Optional — providers may not supply all.
    """

    model_config = ConfigDict(extra="ignore")

    validTime: str                      # UTC ISO-8601 with Z
    outTemp: float | None = None
    outHumidity: float | None = None
    windSpeed: float | None = None
    windDir: float | None = None
    windGust: float | None = None
    precipProbability: float | None = None   # 0-100
    precipAmount: float | None = None
    precipType: str | None = None            # "rain" | "snow" | "freezing-rain" | null
    cloudCover: float | None = None          # 0-100
    weatherCode: str | None = None           # WMO code as string (opaque to api)
    weatherText: str | None = None           # Human-readable (decoded from WMO)
    source: str
    extras: dict[str, Any] = {}


class DailyForecastPoint(BaseModel):
    """Canonical daily forecast record (ADR-010 §3.4, OpenAPI DailyForecastPoint schema).

    extra="ignore" so provider-specific wire fields don't break normalization.
    Required fields per OpenAPI: validDate, source.
    validDate is station-local "YYYY-MM-DD" (ADR-020: date-only fields stay local).
    sunrise/sunset are UTC ISO-8601 with Z (full datetime fields).
    narrative is always None for Open-Meteo (no per-day narrative supplied).
    """

    model_config = ConfigDict(extra="ignore")

    validDate: str                           # station-local "YYYY-MM-DD"
    tempMax: float | None = None
    tempMin: float | None = None
    precipAmount: float | None = None
    precipProbabilityMax: float | None = None  # 0-100
    windSpeedMax: float | None = None
    windGustMax: float | None = None
    sunrise: str | None = None               # UTC ISO-8601 with Z
    sunset: str | None = None                # UTC ISO-8601 with Z
    uvIndexMax: float | None = None
    weatherCode: str | None = None           # WMO code as string
    weatherText: str | None = None
    narrative: str | None = None             # per-day summary (NWS/some Aeris plans; null here)
    source: str
    extras: dict[str, Any] = {}


class ForecastDiscussion(BaseModel):
    """Canonical forecast discussion (ADR-010 §3.5, OpenAPI ForecastDiscussion schema).

    Always null for Open-Meteo (no discussion endpoint).
    Defined here for completeness and for future NWS / Aeris rounds.
    """

    model_config = ConfigDict(extra="ignore")

    headline: str | None = None
    body: str
    issuedAt: str                # UTC ISO-8601 with Z
    validFrom: str | None = None # UTC ISO-8601 with Z
    validUntil: str | None = None
    senderName: str | None = None
    source: str


class ForecastBundle(BaseModel):
    """ForecastBundle container (ADR-010 §3.10, OpenAPI ForecastBundle schema).

    extra="ignore" for round-trip safety through cache (model_dump → model_validate).
    discussion is always None for Open-Meteo; ForecastDiscussion for providers
    that supply it (NWS AFD, some Aeris plans).
    source: provider_id (e.g. "openmeteo") or "none" when no provider is configured.
    """

    model_config = ConfigDict(extra="ignore")

    hourly: list[HourlyForecastPoint] = []
    daily: list[DailyForecastPoint] = []
    discussion: ForecastDiscussion | None = None
    source: str
    generatedAt: str                # UTC ISO-8601 with Z


class ForecastResponse(BaseModel):
    """ForecastResponse envelope (OpenAPI ForecastResponse schema).

    data: ForecastBundle (hourly + daily + discussion + source + generatedAt).
    units: UnitsBlock from services/units.py — same wiring as observations + records.
    source: mirrors data.source (provider_id or "none").
    generatedAt: UTC ISO-8601 with Z.
    """

    data: ForecastBundle
    units: dict[str, str]
    source: str
    generatedAt: str                # UTC ISO-8601 with Z


# ---------------------------------------------------------------------------
# Alerts (ADR-016, canonical-data-model §3.6 + §3.11)
# ---------------------------------------------------------------------------

# ruff: noqa: N815  (field names use NWS/OpenAPI camelCase: senderName, areaDesc, etc.)


class AlertRecord(BaseModel):
    """Canonical alert record (ADR-010 §3.6, OpenAPI AlertRecord schema).

    extra="ignore" so provider wire shapes that have extra fields don't break
    normalization.  Required fields per OpenAPI: id, headline, severity, event,
    effective, source.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    headline: str
    description: str = ""
    severity: str  # enum: advisory | watch | warning
    urgency: str | None = None
    certainty: str | None = None
    event: str
    effective: str  # UTC ISO-8601 with Z
    expires: str | None = None  # UTC ISO-8601 with Z
    senderName: str | None = None
    areaDesc: str | None = None
    category: str | None = None
    source: str


class AlertList(BaseModel):
    """AlertList container (ADR-010 §3.11, OpenAPI AlertList schema)."""

    model_config = ConfigDict(extra="ignore")

    alerts: list[AlertRecord]
    retrievedAt: str  # UTC ISO-8601 with Z
    source: str  # provider_id or "none"


class AlertListResponse(BaseModel):
    """AlertListResponse envelope (OpenAPI AlertListResponse schema)."""

    model_config = ConfigDict(extra="ignore")

    data: AlertList
    source: str  # mirrors data.source
    generatedAt: str  # UTC ISO-8601 with Z


# ---------------------------------------------------------------------------
# AQI (ADR-013, canonical-data-model §3.8)
# ---------------------------------------------------------------------------

# ruff: noqa: N815  (field names use canonical camelCase: aqiCategory, etc.)


class AQIReading(BaseModel):
    """Canonical AQI reading (ADR-010 §3.8, OpenAPI AQIReading schema).

    Raw provider data — no EPA conversion at ingest. The dashboard applies
    any display-scale conversion using aqiScale as the discriminator.
    extra="ignore" so provider wire shapes that have extra fields don't break
    normalization.  Required fields per OpenAPI: observedAt, source.
    All AQI numeric fields are Optional — providers may not supply all.
    """

    model_config = ConfigDict(extra="ignore")

    aqi: float | None = None
    aqiScale: str | None = None          # scale the aqi value is on: "epa" (0-500), "owm" (1-5), etc.
    aqiCategory: str | None = None       # dashboard-computed from aqi+aqiScale; parsers set None
    aqiMainPollutant: str | None = None  # canonical pollutant id: PM2.5/PM10/O3/NO2/SO2/CO
    aqiLocation: str | None = None       # free-form provider location label (PARTIAL-DOMAIN for Open-Meteo)
    pollutantPM25: float | None = None   # µg/m³ (group_concentration)
    pollutantPM10: float | None = None   # µg/m³ (group_concentration)
    pollutantO3: float | None = None     # µg/m³ (group_concentration; raw provider value)
    pollutantNO2: float | None = None    # µg/m³ (group_concentration; raw provider value)
    pollutantSO2: float | None = None    # µg/m³ (group_concentration; raw provider value)
    pollutantCO: float | None = None     # µg/m³ (group_concentration; raw provider value)
    observedAt: str                      # UTC ISO-8601 with Z; required per OpenAPI
    source: str                          # "weewx" (Path A) or provider_id (Path B)


class AQIResponse(BaseModel):
    """AQIResponse envelope (OpenAPI AQIResponse schema).

    data: AQIReading or None (null when no AQI provider configured or no reading).
    units: UnitsBlock — pollutant unit declarations.
    source: provider_id or "none".
    generatedAt: UTC ISO-8601 with Z.
    """

    data: AQIReading | None
    units: dict[str, str]
    source: str
    generatedAt: str  # UTC ISO-8601 with Z


class AQIHistoryResponse(BaseModel):
    """AQIHistoryResponse envelope (OpenAPI AQIHistoryResponse schema, P4-T3).

    data: list of AQIReading from the weewx archive (Path A operators).
    units: UnitsBlock — pollutant unit declarations (same as /aqi/current).
    source: always "weewx" (reads from weewx archive per ADR-013 corrected).
    generatedAt: UTC ISO-8601 with Z.
    page: pagination metadata.
    """

    data: list[AQIReading]
    units: dict[str, str]
    source: str
    generatedAt: str  # UTC ISO-8601 with Z
    page: PageInfo


# ---------------------------------------------------------------------------
# Earthquakes (ADR-040, canonical-data-model §3.7 + §2.4)
# ---------------------------------------------------------------------------

# ruff: noqa: N815  (camelCase canonical names: magnitudeType, etc.)


class EarthquakeRecord(BaseModel):
    """Canonical earthquake record (ADR-010 §3.7, OpenAPI EarthquakeRecord schema).

    extra="ignore" so provider wire shapes that have extra fields don't break
    normalization.  Required fields per OpenAPI: id, time, latitude, longitude,
    magnitude, source.

    Earthquakes are unit-system-invariant (canonical-data-model §2.4) — no
    units block; depth is always km, coordinates always WGS84 degrees.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    time: str  # UTC ISO-8601 with Z
    latitude: float
    longitude: float
    magnitude: float
    magnitudeType: str | None = None
    depth: float | None = None
    place: str | None = None
    url: str | None = None
    tsunami: bool | None = None
    felt: int | None = None
    mmi: float | None = None
    alert: str | None = None  # green/yellow/orange/red (USGS PAGER); None for non-USGS
    status: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)
    source: str


class EarthquakeListResponse(BaseModel):
    """EarthquakeListResponse envelope (OpenAPI EarthquakeListResponse schema).

    Note: OpenAPI EarthquakeListResponse does NOT carry a units block (per
    canonical-data-model §2.4, earthquakes are unit-system-invariant).
    """

    model_config = ConfigDict(extra="ignore")

    data: list[EarthquakeRecord]
    source: str  # provider_id or "none"
    generatedAt: str  # UTC ISO-8601 with Z


# ---------------------------------------------------------------------------
# Radar (ADR-015, 3b-14)
# ---------------------------------------------------------------------------


class RadarFrame(BaseModel):
    """One radar frame (canonical-data-model §4.5, OpenAPI RadarFrame schema).

    Radar has no canonical-entity mapping — tiles are bytes fetched browser-side.
    RadarFrame carries the timestamp, kind, and (RainViewer only) a per-frame
    `path` that the dashboard combines with the response-level `tileHost` to
    materialize the tile URL via CAPABILITY.tile_url_template.

    `path` is None for WMS-T providers (they compose tile URLs purely via
    `?TIME=<time>`); set only for RainViewer per its api-docs (3b-14 auditor F2).
    """

    model_config = ConfigDict(extra="ignore")

    time: str  # UTC ISO-8601 with Z (ADR-020)
    kind: str  # "past" | "current" | "nowcast" (OpenAPI RadarFrame kind enum)
    path: str | None = None  # RainViewer per-frame tile path; None for WMS-T


class RadarFrameList(BaseModel):
    """List of radar frames for a single provider (OpenAPI RadarFrameList schema).

    required: [providerId, frames]
    attribution + tileHost are nullable.

    `tileHost` is RainViewer's per-fetch tile-server host from the JSON envelope
    (`weather-maps.json.host`; currently `https://tilecache.rainviewer.com`).
    The dashboard substitutes it into CAPABILITY.tile_url_template's `{host}`
    placeholder along with each frame's `path`. None for WMS-T providers
    (their tile URL is composed differently — see CAPABILITY.wms_endpoint_url).
    3b-14 auditor F2 added this.
    """

    model_config = ConfigDict(extra="ignore")

    providerId: str
    frames: list[RadarFrame]
    attribution: str | None = None
    tileHost: str | None = None  # RainViewer per-fetch tile host; None for WMS-T


class RadarFramesResponse(BaseModel):
    """RadarFramesResponse envelope (OpenAPI RadarFramesResponse schema).

    Wraps RadarFrameList in the standard data + generatedAt envelope.
    """

    model_config = ConfigDict(extra="ignore")

    data: RadarFrameList
    generatedAt: str  # UTC ISO-8601 with Z
